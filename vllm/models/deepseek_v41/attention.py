# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 attention for SM70 (PORT_DESIGN §3.3; owner L-ATTN). Reference: ref:m.py:613-789.

Per TP rank (TP4: 16 local heads, 2 local o-groups); ``x`` = attn_norm(hc_pre(...)) [T, 5120] fp16:

1. ``qr_kv = [wq_a | wkv] x`` (replicated, FP32 out); ``qr = q_norm(.)`` fp16, ``kv = kv_norm(.)`` FP32.
2. ``q = wq_b(qr)`` [T, 16, 512] fp16, RoPE on dims 448:512 -- **no per-head q-norm**.
3. window KV: RoPE(kv) -> ``fp8_block32_qdq`` over all 512 dims -> FP16 row of ``{P}.swa_cache``.
4. kv sources (2, 8, 14, 20): compressor -> pre-RoPE latent -> ``indexer.write_keys`` (before the cache
   write, ref:m.py:516-548) -> RoPE at the group's first position -> ``fp4_e4m3_qdq`` -> FP16 row of the
   compressed cache ``{P}`` (+ ``shared.export_ckv`` on source 20 of an exporting stage).
5. index sources: ``indexer.forward`` -> ``shared.topk_indices[:T]``; Reuse layers read what their index
   source published earlier in this forward.
6. sparse attention over <= 128 window rows + <= 512 compressed rows of the kv source (logical -> row via
   the SOURCE's block table), ``attn_sink``, running-max floor -1e30, scale 512^-1/2.
7. inverse RoPE on dims 448:512 (1Cat V4 SM70 kernel), grouped ``wo_a`` (FP16 operands, FP32 accumulate),
   ``wo_b`` row-parallel with FP32 output and an FP32 TP all-reduce -> [T, 5120] FP32.

Steps 1-6 run inside the eager-break op ``deepseek_v41_attention`` (host-dependent control flow, cache
writes); step 7 is graph-capturable. Source discovery is by name in the static forward context
(``model.layers.{kv_source}.attn`` / ``….attn.indexer.k_cache``); on a mirrored stage the mirror owns those
names, so consumer code is identical on every stage.
"""

from __future__ import annotations

from typing import cast

import torch
from torch import nn

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.models.deepseek_v41.common.contracts import (
    CAND_SOURCE,
    HEAD_DIM,
    HIDDEN,
    IDX_TOPK,
    N_HEADS,
    O_GROUPS,
    O_LORA,
    Q_LORA,
    ROPE_DIM,
    LayerTopology,
    SharedAttnBuffers,
    StagePlan,
)
from vllm.models.deepseek_v41.common.rope import apply_rope_torch, build_v41_rope
from vllm.models.deepseek_v41.compressor import (
    DeepseekV41Compressor,
    FP32RMSNorm,
    mm_fp32,
)
from vllm.models.deepseek_v41.indexer import DeepseekV41Indexer, attn_impl
from vllm.logger import init_logger
from vllm.models.deepseek_v41.sm70.q_rope_kv_insert import q_rope_kv_insert
from vllm.models.deepseek_v41.sm70.sparse_kernels import (
    ATTN_GRAPH,
    ATTN_OUTER_GRAPH,
    GATHER_ONCE,
    GATHER_ONCE_MIN_T,
)
from vllm.models.deepseek_v41.sm70.gather_once import GroupRowsCache
from vllm.models.deepseek_v41.sm70.sparse import (
    DeepseekV41SM70SparseImpl,
    DS41CacheLayer,
    DS41CompressedBackend,
    DS41CompressedMetadata,
    DS41SWABackend,
    DS41SWAMetadata,
    compressed_cache_spec,
    step_metadata,
    swa_cache_spec,
)
from vllm.models.deepseek_v41.sm70.sparse_kernels import ckv_rope_qat_store
from vllm.models.deepseek_v41.sm70 import fixed_m
from vllm.models.deepseek_v41.sm70 import precapture as _pc

ROPE_MARGIN = 1024          # positions past max_model_len that speculative tokens may reach
_GROUP_ROWS = GroupRowsCache()   # VLLM_DS41_ATTN_GATHER_ONCE: compressed row ids shared within a layer group
# VLLM_DS41_SLICE_SEGMENT_TIMING (layer-sliced prefill): a per-segment CUDA-event timer installed by the slice
# executor around one slice forward (an object with a ``segment(name)`` context manager), else None. Read only on the
# eager prefill path (never in decode graphs / capture).
SEGMENT_TIMER = None


class _NoSegment:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


_NO_SEGMENT = _NoSegment()


def segment(name: str):
    t = SEGMENT_TIMER
    return t.segment(name) if t is not None else _NO_SEGMENT


def _fp16_weight(lin: nn.Module, shape: tuple[int, int], what: str) -> torch.Tensor:
    w = lin.weight
    if w.dtype != torch.float16 or tuple(w.shape) != shape:
        raise RuntimeError(f"{what}: expected an FP16 {shape} weight (dense FP8 is dequantised at load, "
                           f"PORT_DESIGN A2), got {w.dtype} {tuple(w.shape)}")
    return w


class DeepseekV41Attention(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str, topo: LayerTopology, stage: StagePlan,
                 shared: SharedAttnBuffers, aux_streams: list[torch.cuda.Stream] | None = None) -> None:
        super().__init__()
        hf = vllm_config.model_config.hf_config
        tp = get_tensor_model_parallel_world_size()
        if N_HEADS % tp or O_GROUPS % tp:
            raise ValueError(f"TP {tp} must divide {N_HEADS} heads and {O_GROUPS} o-groups")
        self.prefix = prefix
        self.topo = topo
        self.stage = stage
        self.shared = shared
        self.layer_id = topo.layer_id
        self.tp_size = tp
        self.n_local_heads = N_HEADS // tp
        self.n_local_groups = O_GROUPS // tp
        self.scale = HEAD_DIM ** -0.5
        self.eps = float(getattr(hf, "rms_norm_eps", 1e-20))
        block_size = vllm_config.cache_config.block_size
        quant_config = vllm_config.quant_config

        self.fused_wqa_wkv = MergedColumnParallelLinear(
            HIDDEN, [Q_LORA, HEAD_DIM], bias=False, quant_config=quant_config, disable_tp=True,
            return_bias=False, prefix=f"{prefix}.fused_wqa_wkv")
        self.q_norm = FP32RMSNorm(Q_LORA, self.eps)
        self.kv_norm = FP32RMSNorm(HEAD_DIM, self.eps)
        self.wq_b = ColumnParallelLinear(Q_LORA, N_HEADS * HEAD_DIM, bias=False, quant_config=quant_config,
                                         return_bias=False, prefix=f"{prefix}.wq_b")
        self.wo_a = ColumnParallelLinear(N_HEADS * HEAD_DIM // O_GROUPS, O_GROUPS * O_LORA, bias=False,
                                         quant_config=quant_config, return_bias=False, prefix=f"{prefix}.wo_a")
        self.wo_a.is_bmm = True
        self.wo_a.bmm_batch_size = self.n_local_groups
        self.wo_b = RowParallelLinear(O_GROUPS * O_LORA, HIDDEN, bias=False, quant_config=quant_config,
                                      reduce_results=False, return_bias=False, prefix=f"{prefix}.wo_b")
        self.attn_sink = nn.Parameter(torch.zeros(self.n_local_heads, dtype=torch.float32), requires_grad=False)
        set_weight_attrs(self.attn_sink, {"weight_loader": self._load_attn_sink})

        max_pos = int(vllm_config.model_config.max_model_len) + ROPE_MARGIN
        self.__dict__["rotary_emb"] = build_v41_rope(hf, topo.compress_ratio, max_positions=max_pos)

        cfg = vllm_config
        self.swa_cache = DS41CacheLayer(f"{prefix}.swa_cache", swa_cache_spec(block_size), DS41SWABackend, cfg)
        self.compressor: DeepseekV41Compressor | None = None
        self.kv_cache_layer: DS41CacheLayer | None = None
        if topo.owns_compressor:
            self.compressor = DeepseekV41Compressor(cfg, f"{prefix}.compressor", topo.compress_ratio)
            self.kv_cache_layer = DS41CacheLayer(prefix, compressed_cache_spec(block_size, topo.compress_ratio),
                                                 DS41CompressedBackend, cfg)
        self.indexer: DeepseekV41Indexer | None = None
        if topo.owns_indexer:
            self.indexer = DeepseekV41Indexer(cfg, f"{prefix}.indexer", topo, shared)
            self.indexer.__dict__["rotary_emb"] = self.rotary_emb
        if topo.compress_ratio > 0:
            base = prefix[: prefix.rindex(f".{topo.layer_id}.")]
            self.kv_source_name = f"{base}.{topo.kv_source}.attn"
        else:
            self.kv_source_name = None

        self.layer_name = f"{prefix}.ds41_attention"
        ctx = cfg.compilation_config.static_forward_context
        if self.layer_name in ctx:
            raise ValueError(f"Duplicate layer name: {self.layer_name}")
        ctx[self.layer_name] = self
        self._static_ctx = ctx
        # layer-sliced prefill: (i, n) while attention part i of n runs (forward_part), else None
        self._partial: tuple[int, int] | None = None

    # ------------------------------------------------------------------------------- loading
    def _load_attn_sink(self, param: torch.Tensor, loaded: torch.Tensor) -> None:
        if loaded.shape != (N_HEADS,):
            raise ValueError(f"{self.prefix}.attn_sink: checkpoint shape {tuple(loaded.shape)} != ({N_HEADS},)")
        r = get_tensor_model_parallel_rank()
        param.data.copy_(loaded[r * self.n_local_heads:(r + 1) * self.n_local_heads].float())

    # ------------------------------------------------------------------------------- forward
    def forward_part(self, positions: torch.Tensor, x: torch.Tensor, part: tuple[int, int]) -> torch.Tensor | None:
        """Attention part i of n of a prefill chunk (layer-sliced prefill, eager only). Part 0 runs the projections and
        every KV store (SWA, compressor / compressed KV, index keys); parts 1.. recompute the pure projections (and
        re-store the identical SWA rows); the indexer's query-row tiles are split across the parts (each part
        publishes its rows of the shared top-k / candidate blocks); the last part attends and returns the output
        projection. Earlier parts return None."""
        i, n = int(part[0]), int(part[1])
        if not 0 <= i < n:
            raise ValueError(f"{self.prefix}: attention part {i} of {n}")
        T = x.shape[0]
        if T == 0:
            raise ValueError(f"{self.prefix}: attention part of an empty chunk")
        o = torch.empty((T, self.n_local_heads, HEAD_DIM), dtype=torch.float16, device=x.device)
        self._partial = (i, n)
        try:
            deepseek_v41_attention(x, positions, o, self.layer_name)
        finally:
            self._partial = None
        if i < n - 1:
            return None
        with segment("oproj"):
            return self.output_projection(o, positions)

    def _run_partial(self, x, positions, out, md_all, swa_md, T, w_qkv, w_qb, i: int, n: int) -> None:
        st: dict = {}
        with segment("project" if i == 0 else "reproject"):
            self._seg_project_store(x, positions, md_all, swa_md, T, w_qkv, w_qb, st, stores=i == 0)
        if self.topo.compress_ratio > 0 and self.indexer is not None:
            with segment("indexer"):
                self.indexer.forward(x[:T], st["qr32"], positions[:T], tiles=(i, n))
        if i == n - 1:
            with segment("attend"):
                self._seg_attend(out, md_all, swa_md, T, st)

    def forward(self, positions: torch.Tensor, x: torch.Tensor, reduce: bool = True) -> torch.Tensor:
        """``reduce=False`` (sequence-parallel HC, sm70/sp_hc.py): this rank's un-reduced fp32 partial."""
        T = x.shape[0]
        o = torch.empty((T, self.n_local_heads, HEAD_DIM), dtype=torch.float16, device=x.device)
        deepseek_v41_attention(x, positions, o, self.layer_name)
        if SEGMENT_TIMER is not None:
            with segment("oproj"):
                return self.output_projection(o, positions, reduce)
        return self.output_projection(o, positions, reduce)

    def output_projection(self, o: torch.Tensor, positions: torch.Tensor, reduce: bool = True) -> torch.Tensor:
        """inverse RoPE + grouped wo_a + wo_b (FP32 out) + FP32 TP all-reduce (ref:m.py:781-788)."""
        T = o.shape[0]
        cos_sin = self.rotary_emb.cos_sin_cache
        if attn_impl() == "torch":
            o = apply_rope_torch(o, positions[:T], cos_sin, inverse=True)
        else:
            from vllm.models.deepseek_v4.sm70.projection import sm70_inverse_rope
            o = sm70_inverse_rope(o.contiguous(), positions[:T], cos_sin, ROPE_DIM)
        g, d_in = self.n_local_groups, N_HEADS * HEAD_DIM // O_GROUPS
        wo_a = _fp16_weight(self.wo_a, (g * O_LORA, d_in), f"{self.prefix}.wo_a").view(g, O_LORA, d_in)
        og = o.reshape(T, g, d_in).transpose(0, 1)                                   # [g, T, 4096]
        z = torch.bmm(og, wo_a.transpose(1, 2), out_dtype=torch.float32)            # [g, T, 1024]
        z = z.transpose(0, 1).reshape(T, g * O_LORA).to(torch.float16)
        wo_b = _fp16_weight(self.wo_b, (HIDDEN, g * O_LORA), f"{self.prefix}.wo_b")
        out = mm_fp32(z, wo_b)
        if self.tp_size > 1 and reduce:
            out = tensor_model_parallel_all_reduce(out)
        return out

    def attention_impl(self, x: torch.Tensor, positions: torch.Tensor, out: torch.Tensor) -> None:
        md_all = step_metadata(self.layer_name)
        w_qkv = _fp16_weight(self.fused_wqa_wkv, (Q_LORA + HEAD_DIM, HIDDEN), f"{self.prefix}.fused_wqa_wkv")
        w_qb = _fp16_weight(self.wq_b, (self.n_local_heads * HEAD_DIM, Q_LORA), f"{self.prefix}.wq_b")
        capturing = torch.cuda.is_current_stream_capturing()
        if capturing and ATTN_OUTER_GRAPH and self.__dict__.get("_cap_logs", 0) < 40:
            self._cap_logs = self.__dict__.get("_cap_logs", 0) + 1
            logger.info("%s: CAPTURE x=%d rows metadata=%s", self.layer_name, x.shape[0],
                        "none -> PROFILE PATH" if md_all is None else
                        f"T={md_all[self.swa_cache.layer_name].num_actual_tokens} win={md_all[self.swa_cache.layer_name].window_slots.data_ptr():#x} slot={md_all[self.swa_cache.layer_name].slot_mapping.data_ptr():#x} pos={positions.data_ptr():#x} x={x.data_ptr():#x} out={out.data_ptr():#x}")
        if md_all is None:              # dummy/profile forward without metadata (ForwardContext.is_dummy_run)
            self._profile_run(x, positions, out, w_qkv, w_qb)
            return
        swa_md = cast(DS41SWAMetadata, md_all[self.swa_cache.layer_name])
        T = swa_md.num_actual_tokens
        if self._partial is not None:
            self._run_partial(x, positions, out, md_all, swa_md, T, w_qkv, w_qb, *self._partial)
            return
        make_plan = lambda st: self._plan(x, positions, out, md_all, swa_md, T, w_qkv, w_qb, st)
        if (ATTN_GRAPH and LAYER_GRAPHS and 0 < T <= GRAPH_MAX_T and attn_impl() == "sm70"
                and not swa_md.replay_compacted and not torch.cuda.is_current_stream_capturing()):
            self._graph_step(make_plan, x, positions, out, md_all, swa_md, T)
            return
        if SEGMENT_TIMER is not None:          # VLLM_DS41_SLICE_SEGMENT_TIMING (slices only)
            for name, (seg, _) in zip(self._plan_names(), make_plan({})):
                with segment(name):
                    seg()
            return
        for seg, _ in make_plan({}):
            seg()

    def _plan_names(self) -> tuple[str, ...]:
        return (("project", "indexer", "attend") if self.topo.compress_ratio > 0 and self.indexer is not None
                else ("project", "attend"))

    # ---- the attention break in segments (every segment but the full indexer scan takes no host decisions) ----
    def _plan(self, x, positions, out, md_all, swa_md, T, w_qkv, w_qb, st) -> list[tuple]:
        """[(segment, capturable)]: project + store | indexer top-k | gather + attend."""
        plan = [(lambda: self._seg_project_store(x, positions, md_all, swa_md, T, w_qkv, w_qb, st), True)]
        if self.topo.compress_ratio > 0 and self.indexer is not None:
            plan.append((lambda: self.indexer.forward(x[:T], st["qr32"], positions[:T]), self.indexer.graph_capable()))
        plan.append((lambda: self._seg_attend(out, md_all, swa_md, T, st), True))
        return plan

    def _seg_project_store(self, x, positions, md_all, swa_md, T, w_qkv, w_qb, st, stores: bool = True) -> None:
        """q / qr32 into ``st`` + the SWA insert; with ``stores`` (always, except attention parts after the first)
        also the compressor -> index keys + compressed KV stores. The SWA insert writes the same bytes again in later
        parts (same inputs, same kernel)."""
        impl = attn_impl()
        pos = positions[:T]
        qr_kv = mm_fp32(x[:T], w_qkv)
        qr32 = self.q_norm(qr_kv[:, :Q_LORA])           # FP32; the indexer's q GEMM consumes it un-rounded
        qr = qr32.to(torch.float16)
        kv = self.kv_norm(qr_kv[:, Q_LORA:])
        q = fixed_m.mm(qr, w_qb.t()).view(T, self.n_local_heads, HEAD_DIM)
        q_rope_kv_insert(q, kv, pos, self.rotary_emb.cos_sin_cache, self.swa_cache.rows(),
                         swa_md.slot_mapping, impl=impl)
        st["q"], st["qr32"] = q, qr32
        if stores and self.topo.compress_ratio > 0 and self.compressor is not None:
            latent, latent_pos = self.compressor(x[:T], pos)
            assert self.indexer is not None and self.kv_cache_layer is not None
            self.indexer.write_keys(latent, latent_pos)
            src_md = cast(DS41CompressedMetadata, md_all[self.kv_cache_layer.layer_name])
            export = None
            if self.layer_id == CAND_SOURCE and self.shared.export_ckv is not None:
                export = self.shared.export_ckv
            ckv_rope_qat_store(latent, latent_pos, self.rotary_emb.cos_sin_cache, self.kv_cache_layer.rows(),
                               src_md.latent_slots, export, impl=impl)

    def _seg_attend(self, out, md_all, swa_md, T, st) -> None:
        impl = attn_impl()
        ckv_rows = ckv_idx = None
        if self.topo.compress_ratio > 0:
            src_md = cast(DS41CompressedMetadata, md_all[self.kv_source_name])
            src = self._static_ctx.get(self.kv_source_name)
            if src is None:
                raise RuntimeError(f"kv source {self.kv_source_name} is not registered on this stage")
            ckv_rows = cast(DS41CacheLayer, src).rows()
            topk = self.shared.topk_indices[:T]
            if (GATHER_ONCE and impl == "sm70" and T >= GATHER_ONCE_MIN_T
                    and not torch.cuda.is_current_stream_capturing()):
                # prefill: every layer of a (kv source, index source) group selects the same rows (sm70/gather_once.py)
                ckv_idx = _GROUP_ROWS.get(
                    layer_id=self.layer_id, owns_indexer=self.topo.owns_indexer, kv_source=self.topo.kv_source,
                    index_source=self.topo.index_source, src_md=src_md, topk=topk, num_tokens=T,
                    rows_ptr=ckv_rows.data_ptr(),
                    compute=lambda: DeepseekV41SM70SparseImpl.compressed_rows(topk, src_md))
            else:
                ckv_idx = DeepseekV41SM70SparseImpl.compressed_rows(topk, src_md)
        DeepseekV41SM70SparseImpl.forward(st["q"], swa_md, self.swa_cache.rows(), ckv_rows, ckv_idx, self.attn_sink,
                                          self.scale, out[:T], impl=impl)
        if out.shape[0] > T:
            out[T:].zero_()

    # ---- per-layer CUDA graphs of the capturable segments (VLLM_DS41_ATTN_GRAPH) ----
    def _graph_key(self, x, positions, out, md_all, swa_md, T) -> tuple:
        key = [T, out.shape[0], x.data_ptr(), positions.data_ptr(), out.data_ptr(), swa_md.block_table.shape[0],
               swa_md.block_table.data_ptr(), swa_md.window_slots.data_ptr(), swa_md.slot_mapping.data_ptr(),
               swa_md.positions.data_ptr(), swa_md.token_to_req_indices.data_ptr()]
        if self.topo.compress_ratio > 0:
            src_md = cast(DS41CompressedMetadata, md_all[self.kv_source_name])
            key += [src_md.block_table.data_ptr(), src_md.num_visible.data_ptr(), src_md.num_latents]
            if self._regime_keyed():
                key.append(src_md.max_visible <= IDX_TOPK)
            if self.kv_cache_layer is not None:
                own = cast(DS41CompressedMetadata, md_all[self.kv_cache_layer.layer_name])
                key += [own.latent_slots.data_ptr(), own.latent_pos.data_ptr(), own.slot_mapping.data_ptr(),
                        own.num_latents]
        return tuple(key)

    def _regime_keyed(self) -> bool:
        """Whether the visible-entry regime (<= IDX_TOPK compressed entries) is part of this layer's graph key.

        It only changes a CAPTURED segment on layers whose own indexer is graph-capable (reindex layers in candidate
        mode: the regime picks the arange path or the candidate scoring inside the graph). Everywhere else the regime
        only steers the eager indexer segment, and the captured project/store and attend segments are the same
        kernels on the same static buffers in both regimes. VLLM_DS41_ATTN_PRECAPTURE=1 keys it only where it
        matters (half the keys elsewhere); off, every compressed layer keeps the old key."""
        if not _pc.precapture_enabled():
            return True
        return self.indexer is not None and self.indexer.graph_capable()

    def _graph_desc(self, md_all, swa_md, T, out) -> tuple:
        """(T, padded rows, R, regime) of a decode-graph key, for logs: the parts of ``_graph_key`` that vary
        between steps (the pointers are fixed per padded size). regime = 'long' when this layer's kv source has more
        than IDX_TOPK visible compressed entries, '-' for window-only layers, 'any' when the regime is not keyed."""
        regime = "-"
        if self.topo.compress_ratio > 0 and not self._regime_keyed():
            regime = "any"
        elif self.topo.compress_ratio > 0:
            src_md = cast(DS41CompressedMetadata, md_all[self.kv_source_name])
            regime = "short" if src_md.max_visible <= IDX_TOPK else "long"
        return (int(T), int(out.shape[0]), int(swa_md.block_table.shape[0]), regime)

    def _graph_step(self, make_plan, x, positions, out, md_all, swa_md, T) -> None:
        graphs = self.__dict__.setdefault("_decode_graphs", {})
        key = self._graph_key(x, positions, out, md_all, swa_md, T)
        entry = graphs.get(key)
        if (entry is None or entry["graphs"] is None) and not _pc.capture_allowed():
            # VLLM_DS41_ATTN_NO_LAZY_CAPTURE after the startup capture: a key it did not capture runs eagerly for this
            # step (no warm-up, no capture, no new entry), so nothing is captured or pooled mid-serve
            _pc.STATE.eager_steps += 1
            desc = self._graph_desc(md_all, swa_md, T, out)
            if _pc.first_report("eager", desc):
                logger.warning("DS41 attn graph: key T=%d padded=%d R=%d regime=%s was not captured at startup; "
                               "running it eagerly (VLLM_DS41_ATTN_NO_LAZY_CAPTURE=1; logged once per key shape, "
                               "first on %s)", *desc, self.layer_name)
            for seg, _ in make_plan({}):
                seg()
            return
        if entry is None:
            if len(graphs) >= GRAPH_MAX_KEYS:
                if not getattr(self, "_graph_overflow_logged", False):
                    self._graph_overflow_logged = True
                    logger.warning("%s: more than %d decode-graph keys; running the attention break eagerly for "
                                   "new keys", self.layer_name, GRAPH_MAX_KEYS)
                for seg, _ in make_plan({}):
                    seg()
                return
            entry = graphs[key] = {"runs": 0, "graphs": None, "st": {}}
        # the segment state (q, qr32) is the entry's: captured segments rewrite those tensors in place on replay, and
        # the eager / replayed segments after them read the same objects
        plan = make_plan(entry["st"])
        if entry["runs"] < GRAPH_WARMUP_STEPS:      # warm run: Triton JIT, workspaces, allocator state
            entry["runs"] += 1
            for seg, _ in plan:
                seg()
            return
        groups = _group_segments(plan)                # adjacent capturable segments share one graph
        if entry["graphs"] is None:                   # capture step: capture + replay each capturable group
            entry["st"].clear()
            pool = _graph_pool(key)
            # Private graph pools cannot spend the caching allocator reserve. A
            # second live sequence is a new shape (T=12 vs the T=6 startup
            # capture) and is captured in the live step, after a prefill has
            # left that reserve idle and the driver almost empty.
            _release_capture_headroom()
            desc = self._graph_desc(md_all, swa_md, T, out)
            if _pc.STATE.precapturing:
                _pc.STATE.captured.add(desc)
            else:
                seen_t = self.__dict__.setdefault("_graph_capture_ts", set())
                if T not in seen_t:
                    seen_t.add(T)
                    logger.info("%s: capturing %d decode-graph segment group(s) for T=%d", self.layer_name,
                                sum(1 for _, c in groups if c), T)
                if _pc.STATE.done:
                    _pc.STATE.lazy_captures += 1
                    if _pc.first_report("lazy", desc):
                        logger.warning("DS41 attn graph: LAZY capture mid-serve of key T=%d padded=%d R=%d regime=%s "
                                       "(not captured at startup; first on %s)", *desc, self.layer_name)
            reserved0 = torch.cuda.memory_reserved()
            captured = []
            for segs, capturable in groups:
                if capturable:
                    g = _capture_graph(lambda: [s() for s in segs], pool)
                    g.replay()
                    captured.append(g)
                else:
                    for s in segs:
                        s()
                    captured.append(None)
            entry["graphs"] = captured
            _GRAPH_STATS["keys"] += 1
            _GRAPH_STATS["grew"] += torch.cuda.memory_reserved() - reserved0
            _GRAPH_STATS["graphs"] += sum(1 for g in captured if g is not None)
            _GRAPH_STATS["st_bytes"] += sum(t.numel() * t.element_size() for t in entry["st"].values()
                                            if isinstance(t, torch.Tensor))
            if _GRAPH_STATS["keys"] % 20 == 0 and not _pc.STATE.precapturing:
                logger.info("DS41 attn graph: %d keys captured (all layers), allocator reserve grew %.0f MiB "
                            "across captures (shared pool=%s)", _GRAPH_STATS["keys"],
                            _GRAPH_STATS["grew"] / (1024 * 1024), SHARED_POOL)
            return
        for (segs, _), g in zip(groups, entry["graphs"]):
            if g is None:
                for s in segs:
                    s()
            else:
                g.replay()

    def _profile_run(self, x: torch.Tensor, positions: torch.Tensor, out: torch.Tensor,
                     w_qkv: torch.Tensor, w_qb: torch.Tensor) -> None:
        """Dummy/profile forward: run the projections at full size and reserve the attention workspace."""
        qr_kv = mm_fp32(x, w_qkv)
        qr = self.q_norm(qr_kv[:, :Q_LORA]).to(torch.float16)
        torch.mm(qr, w_qb.t())
        if self.compressor is not None:
            self.compressor(x, positions)
        if self.indexer is not None:
            self.indexer.forward(x, qr, positions)
        DeepseekV41SM70SparseImpl.reserve_workspace(out, self.topo.compress_ratio > 0)
        out.zero_()


logger = init_logger(__name__)


GRAPH_MAX_T = 16                 # decode / spec-verify steps only; prefill chunks stay eager
LAYER_GRAPHS = __import__("os").environ.get("VLLM_DS41_ATTN_LAYER_GRAPHS", "1") == "1"   # per-layer graphs of eager-break layers


def _outer_layers() -> frozenset[int] | None:
    """VLLM_DS41_ATTN_OUTER_LAYERS: ids/ranges of the layers captured inline (debug bisection); default all eligible."""
    import os
    spec = os.environ.get("VLLM_DS41_ATTN_OUTER_LAYERS", "all").strip().lower()
    if spec in ("", "all"):
        return None
    if spec == "none":
        return frozenset()
    ids: set[int] = set()
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        ids.update(range(int(lo), int(hi or lo) + 1))
    return frozenset(ids)


_OUTER_LAYERS = _outer_layers()
GRAPH_WARMUP_STEPS = 1
GRAPH_MAX_KEYS = 32
_GRAPH_POOLS: dict = {}
_GRAPH_STATS = {"keys": 0, "grew": 0, "graphs": 0, "st_bytes": 0}   # st_bytes: per-key q/qr32 held in the pool
_GRAPH_STREAM: list = []


def ensure_graph_key_capacity(n: int) -> int:
    """Raise the per-layer decode-graph key cap to at least ``n`` (the startup capture plans more keys than the
    lazy default of 32). Returns the cap."""
    global GRAPH_MAX_KEYS
    GRAPH_MAX_KEYS = max(GRAPH_MAX_KEYS, int(n))
    return GRAPH_MAX_KEYS


def layer_graphs_active() -> tuple[bool, str]:
    """Whether ``_graph_step`` runs at all (else a startup capture has nothing to capture), with the reason."""
    if not ATTN_GRAPH:
        return False, "VLLM_DS41_ATTN_GRAPH=0"
    if not LAYER_GRAPHS:
        return False, "VLLM_DS41_ATTN_LAYER_GRAPHS=0"
    if attn_impl() != "sm70":
        return False, f"attention impl {attn_impl()!r} != 'sm70'"
    return True, ""


def graph_stats() -> dict:
    return dict(_GRAPH_STATS)


def _group_segments(plan) -> list[tuple[list, bool]]:
    groups: list[tuple[list, bool]] = []
    for seg, capturable in plan:
        if groups and groups[-1][1] == capturable:
            groups[-1][0].append(seg)
        else:
            groups.append(([seg], capturable))
    return groups


SHARED_POOL = __import__("os").environ.get("VLLM_DS41_ATTN_SHARED_POOL", "1") == "1"


def _graph_pool(key: tuple):
    """The private allocator pool for a decode-graph key.

    VLLM_DS41_ATTN_SHARED_POOL=1 (default): one pool for every key. A pool per key kept each key's capture
    high-water reserved forever (956 MiB/GPU after a day, unprofiled; the 2026-10-23 21:16 OOM). Sharing is safe
    here because nothing a graph allocates is read after another graph replays, except the entry's own ``st``
    (q, qr32), which the entry keeps referenced, so no later capture is handed its blocks. Temporaries only
    matter during their own replay, and replays are serial on one stream. =0 restores a pool per key."""
    if SHARED_POOL:
        key = ()
    pool = _GRAPH_POOLS.get(key)
    if pool is None:
        pool = _GRAPH_POOLS[key] = torch.cuda.graph_pool_handle()
    return pool



def _release_capture_headroom() -> None:
    """Return idle caching-allocator reserve to the driver before a new capture.

    Replay never calls this. Each rank syncs only its own device, and only when
    that device has under 256 MiB free and at least 32 MiB of reserved idle
    memory. There is no collective here.
    """
    if torch.cuda.is_current_stream_capturing():
        return
    free, _total = torch.cuda.mem_get_info()
    if free >= 256 * 1024 * 1024:
        return
    idle = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
    if idle < 32 * 1024 * 1024:
        return
    torch.cuda.empty_cache()
    free_after, _total = torch.cuda.mem_get_info()
    logger.warning(
        "DS41 attn graph: %.0f MiB free before a new decode-graph capture; "
        "empty_cache returned %.0f MiB (reserved idle %.0f MiB).",
        free / (1024 * 1024),
        (free_after - free) / (1024 * 1024),
        idle / (1024 * 1024),
    )


def _capture_graph(fn, pool) -> torch.cuda.CUDAGraph:
    """Capture ``fn`` into a CUDA graph on a side stream (no sync / gc / empty_cache, unlike torch.cuda.graph):
    the current stream's work is ordered before the capture and the capture's completion before what follows."""
    if not _GRAPH_STREAM:
        _GRAPH_STREAM.append(torch.cuda.Stream())
    side = _GRAPH_STREAM[0]
    cur = torch.cuda.current_stream()
    side.wait_stream(cur)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.stream(side):
        g.capture_begin(pool=pool, capture_error_mode="thread_local")
        try:
            fn()
        finally:
            g.capture_end()
    cur.wait_stream(side)
    return g


def _attention_op(x: torch.Tensor, positions: torch.Tensor, out: torch.Tensor, layer_name: str) -> None:
    forward_context: ForwardContext = get_forward_context()
    layer = forward_context.no_compile_layers[layer_name]
    layer.attention_impl(x, positions, out)


_attention_eager_break = eager_break_during_capture(_attention_op)


def deepseek_v41_attention(x: torch.Tensor, positions: torch.Tensor, out: torch.Tensor, layer_name: str) -> None:
    """The attention op: an eager break of the outer breakable graph, except (ATTN_OUTER_GRAPH) for decode-sized
    batches of layers without an indexer scan, which run inline and are captured with the rest of the layer."""
    if ATTN_OUTER_GRAPH and x.shape[0] <= GRAPH_MAX_T:
        ctx = get_forward_context()
        layer = ctx.no_compile_layers[layer_name]
        # inline only when this forward carries attention metadata: a capture pass without it (the profile stub) must
        # stay an eager break, or the graph would replay zero attention for this layer
        if (layer.indexer is None and (_OUTER_LAYERS is None or layer.layer_id in _OUTER_LAYERS)
                and isinstance(ctx.attn_metadata, dict)):
            _attention_op(x, positions, out, layer_name)
            return
    _attention_eager_break(x, positions, out, layer_name)

