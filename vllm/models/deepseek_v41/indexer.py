# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 lightning indexer (PORT_DESIGN §1 "Indexer" + "Candidate block mask", §3.3; owner L-ATTN).

Constructed on index sources (2, 8, 14, 20 = kv sources; 24, 28, 32, 36 = Reindex). Reference
ref:m.py:488-580. Replicated on every TP rank (as 1Cat V4): all 32 heads, no score all-reduce, so the
top-k is bitwise identical across ranks given identical inputs.

* ``write_keys`` (kv sources): ``k = fp4_e8m0_qdq(rope(k_norm(wk(latent))))`` -> index-K cache rows (and,
  on source 20 of a stage exporting it, ``shared.export_ik``). K comes from the PRE-RoPE compressed latent.
* ``forward``: ``q = fp4_e8m0_qdq(rope(wq_b(qr)))`` [T, 32, 128]; ``w = weights_proj(x) * 128^-1/2 * 32^-1/2``;
  ``score[t, j] = sum_h relu(q[t,h] . k[j]) * w[t,h]`` over the kv source's visible entries
  ``j < (p_t + 1) // ratio`` (FP32); layer 20 publishes candidate blocks (``common.candidate_blocks``),
  24/28/32/36 mask to them; top-``min(512, n)`` re-sorted by position -> ``shared.topk_indices[:T]``.
  When no query of the batch sees more than 512 entries the result is all visible positions and no
  score is computed (layer 20 then publishes ``CAND_ALL``: <= 16,384 positions is a no-op mask).
"""

from __future__ import annotations

from typing import cast

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.models.deepseek_v41 import knobs
from vllm.models.deepseek_v41.sm70.sparse_kernels import ATTN_GRAPH
from vllm.models.deepseek_v41.common.candidate_blocks import (
    apply_candidate_mask,
    select_candidate_blocks,
)
from vllm.models.deepseek_v41.common.contracts import (
    CAND_ALL,
    CAND_BLOCK,
    CAND_SOURCE,
    CAND_TOPK_BLOCKS,
    CKV_RECORD_DIM,
    HIDDEN,
    IDX_DIM,
    IDX_HEADS,
    IDX_TOPK,
    Q_LORA,
    LayerTopology,
    SharedAttnBuffers,
)
from vllm.models.deepseek_v41.compressor import FP32RMSNorm, mm_fp32, mm_fp32_full
from vllm.models.deepseek_v41.sm70.indexer_kernels import (
    index_k_rope_qat_store,
    index_q_rope_qat,
    index_scores,
    topk_sorted,
)
from vllm.models.deepseek_v41.sm70.sparse import (
    DS41CacheLayer,
    DS41CompressedBackend,
    DS41CompressedMetadata,
    index_k_cache_spec,
    step_metadata,
)
from vllm.models.deepseek_v41.sm70.sparse_kernels import logical_to_rows
from vllm.models.deepseek_v41.sm70 import fixed_m
from vllm.models.deepseek_v41.sm70 import indexer_fused

CANDIDATE_SCORING_ENV = "VLLM_DS41_INDEXER_CANDIDATE_SCORING"
SCORE_BUDGET_ENV = "VLLM_DS41_ATTN_INDEXER_SCORE_MIB"
# decode-sized steps (T <= DECODE_ROWS_MAX_T, the per-layer graph range): candidate scoring rows per tile. 0 = the
# score-budget tile (16 rows: one tile per decode step). Each row holds ~6 MiB of scratch (16,384 keys x 128 fp16 +
# 32 x 16,384 fp32 logits); in a captured graph that scratch stays in the private pool, so the T=12 verify alone
# held ~80 MiB per GPU. Prefill tiling is unchanged.
CAND_DECODE_ROWS_ENV = "VLLM_DS41_ATTN_CAND_DECODE_ROWS"
DECODE_ROWS_MAX_T = 16
# VLLM_DS41_INDEXER_CAND_PREFILL: how reindex layers (24-36) with candidate scoring score PREFILL steps (T > 16, not
# capturing). off = today (byte-for-byte); scan = the reference's full scan + candidate mask (layer 20's kernels, cost
# ~ T x n); auto = per request by its visible count n alone: scan when n < VLLM_DS41_INDEXER_CAND_SCAN_MAX, else
# today's candidate_topk (CAND_ALL rows expanded as on the device path). Deterministic per (n, T).
# GPU (T=1024, 20261024T211909Z): cand 65.9 ms flat; scan+mask 12.0 / 23.9 / 79.9 / 295 ms at n = 8K / 17K / 64K /
# 256K (~2.5 + 1.12 ms per 1K keys; both linear in T for T >= 128) -> crossover ~57K, default 48K; identical rows.
CAND_PREFILL_ENV = "VLLM_DS41_INDEXER_CAND_PREFILL"
CAND_PREFILL_MODES = ("off", "scan", "auto")
CAND_SCAN_MAX_ENV = "VLLM_DS41_INDEXER_CAND_SCAN_MAX"               # auto: full scan below this n (default 49152)


def attn_impl() -> str:
    return knobs.env_str("VLLM_DS41_ATTN_IMPL", "sm70", choices=("sm70", "torch"))


def candidate_topk(q: torch.Tensor, w: torch.Tensor, cand: torch.Tensor, ends: torch.Tensor,
                   keys_all: torch.Tensor, rows_of) -> torch.Tensor:
    """Top-512 index positions over the candidate blocks only (reindex layers 24-36).

    q [Q, 32, 128] fp16 (QAT'd, rotated), w [Q, 32] fp32, cand [Q, 2048] int32 candidate block ids (ascending, -1
    pad; no CAND_ALL rows), ends [Q] visible count, keys_all [rows, 128] fp16, ``rows_of(pos [Q, K]) -> rows [Q, K]``
    (-1 for pos -1). Returns [Q, 512] int32 logical positions ascending, -1 padding at the end -- the same contract as
    the full-scan path (``apply_candidate_mask`` + ``topk_sorted``), whose scores it reproduces up to GEMM-kernel
    rounding: the dot products are the same 128-term products, computed here as one bmm per token tile."""
    Q = q.shape[0]
    dev = q.device
    pos = (cand.to(torch.int64)[:, :, None] * CAND_BLOCK + torch.arange(CAND_BLOCK, device=dev)[None, None, :]).view(Q, -1)
    valid = (cand >= 0)[:, :, None].expand(-1, -1, CAND_BLOCK).reshape(Q, -1) & (pos < ends[:, None])
    pos = torch.where(valid, pos, torch.full_like(pos, -1))
    rows = rows_of(pos)                                                                  # [Q, K] int64, -1 invalid
    keys = keys_all.index_select(0, rows.clamp(min=0).view(-1)).view(Q, -1, keys_all.shape[1])   # [Q, K, 128]
    logits = torch.bmm(q, keys.transpose(1, 2), out_dtype=torch.float32)                # [Q, 32, K]
    del keys                         # free the gathered keys before the weighting (same values, lower peak)
    # in place: the same elementwise products as ``relu(logits) * w`` without a second [Q, 32, K] fp32 tensor
    scores = logits.relu_().mul_(w.float()[:, :, None]).sum(dim=1)                       # [Q, K]
    del logits
    scores.masked_fill_(~valid, float("-inf"))
    k = min(IDX_TOPK, scores.shape[1])
    vals, idx = scores.topk(k, dim=-1, sorted=False)
    picked = pos.gather(1, idx)
    picked = torch.where(vals > float("-inf"), picked, torch.full_like(picked, torch.iinfo(torch.int64).max))
    picked = picked.sort(dim=-1).values                                                  # ascending, invalid last
    picked = torch.where(picked == torch.iinfo(torch.int64).max, torch.full_like(picked, -1), picked)
    out = torch.full((Q, IDX_TOPK), -1, dtype=torch.int32, device=dev)
    out[:, :k] = picked.to(torch.int32)
    return out


def _segment(name: str):
    """VLLM_DS41_SLICE_SEGMENT_TIMING: attention.segment(name) (looked up lazily: attention imports this module)."""
    import sys

    mod = sys.modules.get("vllm.models.deepseek_v41.attention")
    return mod.segment(name) if mod is not None else _NullCtx()


class _NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class DeepseekV41Indexer(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str, topo: LayerTopology,
                 shared: SharedAttnBuffers) -> None:
        super().__init__()
        if not topo.owns_indexer:
            raise ValueError(f"layer {topo.layer_id} is not an index source")
        if not prefix.endswith(".indexer"):
            raise ValueError(f"indexer prefix must end with '.indexer', got {prefix!r}")
        self.prefix = prefix
        self.topo = topo
        self.shared = shared
        self.layer_id = topo.layer_id
        self.ratio = topo.compress_ratio
        attn_prefix = prefix[: -len(".indexer")]
        base = attn_prefix[: attn_prefix.rindex(f".{topo.layer_id}.")]          # "model.layers"
        self.k_cache_name = f"{base}.{topo.kv_source}.attn.indexer.k_cache"
        quant_config = vllm_config.quant_config
        self.wq_b = ReplicatedLinear(Q_LORA, IDX_HEADS * IDX_DIM, bias=False, quant_config=quant_config,
                                     return_bias=False, prefix=f"{prefix}.wq_b")
        self.weights_proj = ReplicatedLinear(HIDDEN, IDX_HEADS, bias=False, quant_config=None,
                                             return_bias=False, prefix=f"{prefix}.weights_proj")
        self.softmax_scale = IDX_DIM ** -0.5
        self.owns_k = topo.owns_compressor
        if self.owns_k:
            self.wk = ReplicatedLinear(CKV_RECORD_DIM, IDX_DIM, bias=False, quant_config=None,
                                       return_bias=False, prefix=f"{prefix}.wk")
            self.k_norm = FP32RMSNorm(IDX_DIM)
            self.k_cache = DS41CacheLayer(self.k_cache_name, index_k_cache_spec(
                vllm_config.cache_config.block_size, self.ratio), DS41CompressedBackend, vllm_config)
        # shared with (and attached by) the attention layer; kept out of the module tree on purpose
        self.__dict__["rotary_emb"] = None
        self.score_budget = knobs.env_int(SCORE_BUDGET_ENV, 128, minimum=8) << 20
        self.cand_decode_rows = knobs.env_int(CAND_DECODE_ROWS_ENV, 0, minimum=0)
        # reindex layers (24-36): score only the candidate blocks' keys instead of all n
        self.candidate_scoring = knobs.env_bool(CANDIDATE_SCORING_ENV, False)
        # VLLM_DS41_INDEXER_QTILE_MAX: query-row tile cap of the full scan (0 = none); on by default with layer-sliced
        # prefill's stage-1 attention parts, which split the scan's tiles across slices (same grid sliced / unsliced)
        from vllm.v1 import ds41_slices

        self.qtile_cap = ds41_slices.indexer_qtile_max()
        self.cand_prefill = knobs.env_str(CAND_PREFILL_ENV, "off", choices=CAND_PREFILL_MODES)
        # default follows VLLM_DS41_INDEXER_FUSED (the scan / candidate crossover moves with the fused kernels)
        self.cand_scan_max = ds41_slices.cand_scan_max()
        # VLLM_DS41_INDEXER_FUSED: prefill-sized steps score + select with the fused SM70 kernels (sm70/indexer_fused)
        self.fused = indexer_fused.fused_enabled()
        self.topk_v2 = indexer_fused.topk_v2_enabled()        # VLLM_DS41_INDEXER_TOPK_V2 (scan path, fused only)
        # VLLM_DS41_INDEXER_FUSED_DECODE: decode-sized steps (T <= 24) score + select with the graph-capturable fused
        # kernels on every index source (reindex: candidates; 2 / 8 / 14 / 20: full scan via block maxima), so the
        # whole indexer is captured in the per-layer decode graphs. Scratch is allocated lazily at the first eager
        # decode call (after KV sizing, from the precapture reserve), never here: see indexer_fused.decode_buffers.
        self.fused_decode = indexer_fused.fused_decode_enabled()
        self._dec_max_n = int(vllm_config.model_config.max_model_len) + 1
        self._static_ctx = vllm_config.compilation_config.static_forward_context

    # ---------------------------------------------------------------------------------- helpers
    def _rope(self) -> torch.Tensor:
        if self.rotary_emb is None:
            raise RuntimeError(f"{self.prefix}: rotary_emb not attached by DeepseekV41Attention")
        return self.rotary_emb.cos_sin_cache

    def _metadata(self) -> DS41CompressedMetadata | None:
        md = step_metadata(self.k_cache_name)
        if md is None:                  # dummy/profile forward without metadata (ForwardContext.is_dummy_run)
            return None
        m = md.get(self.k_cache_name)
        if m is None:
            raise RuntimeError(f"no metadata for {self.k_cache_name}")
        return cast(DS41CompressedMetadata, m)

    def _key_rows(self) -> torch.Tensor:
        layer = self._static_ctx.get(self.k_cache_name)
        if layer is None:
            raise RuntimeError(f"index-K source {self.k_cache_name} is not registered on this stage")
        return cast(DS41CacheLayer, layer).rows()

    @staticmethod
    def _w(lin: nn.Module, shape: tuple[int, int]) -> torch.Tensor:
        w = lin.weight
        if w.dtype != torch.float16 or tuple(w.shape) != shape:
            raise RuntimeError(f"indexer weight must be FP16 {shape}, got {w.dtype} {tuple(w.shape)}")
        return w

    # ---------------------------------------------------------------------------------- K
    def write_keys(self, latent: torch.Tensor, latent_pos: torch.Tensor) -> None:
        if not self.owns_k:
            raise RuntimeError(f"layer {self.layer_id} does not own index keys")
        md = self._metadata()
        wk = self._w(self.wk, (IDX_DIM, CKV_RECORD_DIM))
        if md is None:
            return
        n = latent.shape[0]
        if n != md.num_latents:
            raise RuntimeError(f"{self.prefix}: {n} latents but metadata expects {md.num_latents}")
        # [N, 512] x [512, 128] in FP32 (negligible cost): no FP16 rounding of the latent before the FP4 QAT,
        # which otherwise flips ~0.7 % of index-K records vs the v100-semantic reference
        k = self.k_norm(fixed_m.mm(latent.float(), wk.float().t()))
        export = None
        if self.layer_id == CAND_SOURCE and self.shared.export_ik is not None:
            export = self.shared.export_ik
        index_k_rope_qat_store(k, latent_pos, self._rope(), self._key_rows(), md.latent_slots, export,
                               impl=attn_impl())

    # ---------------------------------------------------------------------------------- top-k
    def graph_capable(self) -> bool:
        """True when ``forward`` takes no host decisions on decode-sized steps (reindex layers with candidate scoring
        in graph mode; with VLLM_DS41_INDEXER_FUSED_DECODE every index source)."""
        return ATTN_GRAPH and (self._cand_device() or bool(self.__dict__.get("fused_decode", False)))

    def _cand_device(self) -> bool:
        """Reindex layers' device candidate path (any T, graph mode): today's rule."""
        return ATTN_GRAPH and self.candidate_scoring and self.topo.uses_candidates

    def fused_decode_active(self, T: int) -> bool:
        return bool(self.__dict__.get("fused_decode", False)) and 0 < T <= indexer_fused.DECODE_MAX_T

    def _forward_fused_decode(self, x: torch.Tensor, qr: torch.Tensor, positions: torch.Tensor, md,
                              topk_out: torch.Tensor, cand_out: torch.Tensor) -> None:
        """VLLM_DS41_INDEXER_FUSED_DECODE: device-only, graph-capturable. Reindex layers: top-512 over the published
        candidate blocks (CAND_ALL in-kernel); index sources: full scan via per-block maxima -> top-2048 blocks (layer
        20 publishes them as its candidate blocks) -> top-512 positions. Tie rule: lowest position."""
        T = md.num_actual_tokens
        impl = attn_impl()
        q = mm_fp32_full(qr[:T], self._w(self.wq_b, (IDX_HEADS * IDX_DIM, Q_LORA)))
        q = index_q_rope_qat(q.view(T, IDX_HEADS, IDX_DIM), positions[:T], self._rope(), impl=impl)
        w = mm_fp32(x[:T], self._w(self.weights_proj, (IDX_HEADS, HIDDEN))) * (
            self.softmax_scale * IDX_HEADS ** -0.5)
        keys_all = self._key_rows()
        bufs = indexer_fused.decode_buffers(keys_all.device, self.__dict__.get("_dec_max_n", 0))
        args = (md.num_visible[:T], keys_all, md.token_to_req_indices[:T], md.block_table, md.storage_block_size)
        if self.topo.uses_candidates:
            indexer_fused.fused_decode_candidates(q, w, cand_out[:T], *args, topk_out, bufs)
        else:
            indexer_fused.fused_decode_scan(q, w, *args, topk_out,
                                            cand_out[:T] if self.topo.is_candidate_source else None, bufs)

    def candidate_tile(self, T: int) -> int:
        ctile = max(1, self.score_budget // (CAND_TOPK_BLOCKS * CAND_BLOCK * 2 * IDX_DIM * 2))
        if self.cand_decode_rows and T <= DECODE_ROWS_MAX_T:
            ctile = min(ctile, self.cand_decode_rows)
        return ctile
    def fused_active(self, T: int) -> bool:
        """VLLM_DS41_INDEXER_FUSED for a step of T rows: prefill-sized (T > 16) and not capturing (decode and graph
        capture keep today's kernels)."""
        return (bool(self.__dict__.get("fused", False)) and T > DECODE_ROWS_MAX_T
                and not (torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()))

    def cand_prefill_mode(self, T: int) -> str:
        """VLLM_DS41_INDEXER_CAND_PREFILL for a step of T rows: "off" unless this is a reindex layer in candidate mode,
        T is prefill-sized and nothing is being captured (the other modes take host decisions)."""
        mode = self.__dict__.get("cand_prefill", "off")
        if (mode == "off" or not (self.candidate_scoring and self.topo.uses_candidates) or T <= DECODE_ROWS_MAX_T
                or (torch.cuda.is_available() and torch.cuda.is_current_stream_capturing())):
            return "off"
        return mode

    def cand_kind(self, mode: str, n: int, cand_rows: torch.Tensor) -> str:
        """Per request: "cand" (candidate_topk tiles) or "scan" (full scan + mask). off = today's rule (candidates when
        no row is CAND_ALL); scan = always the scan; auto = by n alone (deterministic per (n, T))."""
        if not (self.candidate_scoring and self.topo.uses_candidates):
            return "scan"
        if mode == "scan":
            return "scan"
        if mode == "auto":
            return "scan" if n < self.cand_scan_max else "cand"
        return "scan" if bool((cand_rows[:, 0] == CAND_ALL).any()) else "cand"

    def _forward_candidates_device(self, x: torch.Tensor, qr: torch.Tensor, positions: torch.Tensor, md,
                                   topk_out: torch.Tensor, cand_out: torch.Tensor,
                                   tiles: tuple[int, int] | None = None) -> None:
        """Reindex layers (24-36) without host decisions: the candidate list is the published blocks, or blocks
        0..2047 when layer 20 published CAND_ALL (<= 16,384 visible entries, which 2048 x 8 covers exactly); then the
        top-512 over the candidates' keys (``candidate_topk``). ``num_visible`` masks positions >= n, so a token
        with nothing visible gets all -1."""
        T = md.num_actual_tokens
        impl = attn_impl()
        q = mm_fp32_full(qr[:T], self._w(self.wq_b, (IDX_HEADS * IDX_DIM, Q_LORA)))
        q = index_q_rope_qat(q.view(T, IDX_HEADS, IDX_DIM), positions[:T], self._rope(), impl=impl)
        w = mm_fp32(x[:T], self._w(self.weights_proj, (IDX_HEADS, HIDDEN))) * (
            self.softmax_scale * IDX_HEADS ** -0.5)
        blocks = torch.arange(CAND_TOPK_BLOCKS, device=x.device, dtype=cand_out.dtype)[None, :]
        cand = torch.where(cand_out[:, :1] == CAND_ALL, blocks, cand_out)
        keys_all = self._key_rows()
        ctile = self.candidate_tile(T)
        starts = list(range(0, T, ctile))
        if tiles is not None:                  # layer-sliced prefill attention part: its share of the row tiles
            from vllm.v1.ds41_slices import split_range

            lo, hi = split_range(len(starts), *tiles)
            starts = starts[lo:hi]
        if starts and self.fused_active(T):     # one fused call over the part's rows (results are per row)
            a, b = starts[0], min(T, starts[-1] + ctile)
            topk_out[a:b] = indexer_fused.fused_candidate_topk(
                q[a:b], w[a:b], cand_out[a:b], md.num_visible[a:b], keys_all, md.token_to_req_indices[a:b],
                md.block_table, md.storage_block_size)
            return
        for a in starts:
            b = min(T, a + ctile)
            topk_out[a:b] = candidate_topk(
                q[a:b], w[a:b], cand[a:b], md.num_visible[a:b], keys_all,
                lambda pos: logical_to_rows(pos, md.token_to_req_indices[a:b], md.block_table,
                                            md.storage_block_size))

    def scan_tasks(self, md, cand_out: torch.Tensor, mode: str = "off",
                   capped: bool = True) -> tuple[list[tuple[str, int, int, int]], dict]:
        """The full-scan work of a step as ordered tasks (kind, request, row a, row b): "empty" (no visible entry),
        "cand" (candidate scoring, reindex layers) or "scan" (one query-row tile of the full scan), plus per-request
        (t0, t1, n, key_tile). The tile grid depends only on the request (n, its rows) and VLLM_DS41_INDEXER_QTILE_MAX;
        the key tile only on the uncapped query tile (the cap changes M of the scan GEMMs, nothing else). ``mode`` =
        VLLM_DS41_INDEXER_CAND_PREFILL for the step (cand_prefill_mode): which requests of a reindex layer score their
        candidates ("cand") and which run the full scan + candidate mask ("scan"; cand_kind)."""
        qsl = md.query_start_loc_cpu
        pos_cpu = md.positions_cpu
        tasks: list[tuple[str, int, int, int]] = []
        info: dict[int, tuple[int, int, int, int]] = {}
        for r in range(md.num_reqs):
            t0, t1 = int(qsl[r]), int(qsl[r + 1])
            if t1 <= t0:
                continue
            n = int((pos_cpu[t1 - 1] + 1) // self.ratio)
            if n == 0:
                tasks.append(("empty", r, t0, t1))
                continue
            q_tile = max(1, min(t1 - t0, self.score_budget // (4 * n)))
            key_tile = max(1024, self.score_budget // (4 * IDX_HEADS * q_tile))
            if self.qtile_cap and capped:
                q_tile = min(q_tile, self.qtile_cap)
            info[r] = (t0, t1, n, key_tile)
            if self.cand_kind(mode, n, cand_out[t0:t1]) == "cand":       # VLLM_DS41_INDEXER_CAND_PREFILL
                ctile = max(1, self.score_budget // (CAND_TOPK_BLOCKS * CAND_BLOCK * 2 * IDX_DIM * 2))
                tasks.extend(("cand", r, a, min(t1, a + ctile)) for a in range(t0, t1, ctile))
            else:
                tasks.extend(("scan", r, a, min(t1, a + q_tile)) for a in range(t0, t1, q_tile))
        return tasks, info

    def _write_empty(self, topk_out: torch.Tensor, cand_out: torch.Tensor, a: int, b: int) -> None:
        topk_out[a:b] = -1
        if self.topo.is_candidate_source:
            cand_out[a:b] = -1
            cand_out[a:b, 0] = CAND_ALL

    def forward(self, x: torch.Tensor, qr: torch.Tensor, positions: torch.Tensor,
                tiles: tuple[int, int] | None = None) -> None:
        """x [T,5120] fp16; qr [T,1280] = q_norm(wq_a x), FP32 preferred (fp16 accepted: PORT_DESIGN §3.3 says fp16 --
        DESIGN-CHANGE REQUEST in L-ATTN.progress.md); positions [T] int64. Writes shared.topk_indices[:T] (and
        shared.candidate_blocks[:T] on layer 20). ``tiles`` = (i, n) (layer-sliced prefill attention parts): only
        part i of the step's ordered scan tasks runs (the early regime / graph path run whole in part 0)."""
        md = self._metadata()
        if md is None:
            mm_fp32(x, self._w(self.weights_proj, (IDX_HEADS, HIDDEN)))
            return
        T = md.num_actual_tokens
        topk_out = self.shared.topk_indices[:T]
        cand_out = self.shared.candidate_blocks[:T]
        if T == 0:
            return
        if tiles is not None and tiles[0] != 0 and md.max_visible <= IDX_TOPK:
            return
        if md.max_visible <= IDX_TOPK:
            # every visible entry is selected: [0, 1, ..., n_t - 1, -1, ...]  (device-only; in graph mode this regime
            # is part of the attention graph key, so a captured graph never crosses it)
            ar = torch.arange(IDX_TOPK, device=x.device, dtype=torch.int64)[None, :]
            topk_out.copy_(torch.where(ar < md.num_visible[:, None], ar, torch.full_like(ar, -1)).to(torch.int32))
            if self.topo.is_candidate_source:
                cand_out.fill_(-1)
                cand_out[:, 0] = CAND_ALL
            return
        # Top-512 dispatch order (rivet/final-1025): (1) the early regime (max_visible <= 512: arange, above);
        # (2) VLLM_DS41_INDEXER_FUSED_DECODE for T <= 24 rows, never attention parts -- its own topk_sel, graph-
        # capturable; (3) the reindex device candidate path (today); (4) the host task path, where for T > 16 the
        # scan tasks use INDEXER_FUSED's dense kernel and, with VLLM_DS41_INDEXER_TOPK_V2, topk_sel with the causal
        # bound in-kernel (else topk512 / topk_sorted). So with both knobs on, steps of 17..24 rows take (2).
        if tiles is None and self.fused_decode_active(T):
            with _segment("idx_fused_decode"):
                self._forward_fused_decode(x, qr, positions, md, topk_out, cand_out)
            return
        mode = self.cand_prefill_mode(T)
        if self._cand_device() and mode == "off":
            with _segment("idx_cand"):
                self._forward_candidates_device(x, qr, positions, md, topk_out, cand_out, tiles)
            return
        impl = attn_impl()
        # the tile cap only inside a staged slice (attention parts, their verify re-run): normal-path prefill keeps
        # today's grid even on a sliced server
        from vllm.v1 import ds41_slices

        tasks, info = self.scan_tasks(md, cand_out, mode,
                                      capped=tiles is not None or ds41_slices.slice_forward_active())
        if tiles is not None:
            from vllm.v1.ds41_slices import split_range

            lo, hi = split_range(len(tasks), *tiles)
            tasks = tasks[lo:hi]
        fused = self.fused_active(T)
        if fused:                               # adjacent candidate tiles of one request (in this part) -> one call
            merged: list[tuple[str, int, int, int]] = []
            for task in tasks:
                if (merged and task[0] == "cand" and merged[-1][0] == "cand" and merged[-1][1] == task[1]
                        and merged[-1][3] == task[2]):
                    merged[-1] = ("cand", task[1], merged[-1][2], task[3])
                else:
                    merged.append(task)
            tasks = merged
        if not any(kind != "empty" for kind, *_ in tasks):
            for kind, r, a, b in tasks:
                self._write_empty(topk_out, cand_out, a, b)
            return
        # qr arrives in FP32 and the GEMM output stays FP32, so the only rounding before the FP4 QAT is the QAT
        # itself: FP16 rounding of qr or of q flips QAT values and changes near-tie top-512 picks (synthetic
        # layer 20: 60 % -> 99.5 % of tokens with the reference's exact top-512 set)
        with _segment("idx_proj"):
            q = mm_fp32_full(qr[:T], self._w(self.wq_b, (IDX_HEADS * IDX_DIM, Q_LORA)))
            q = index_q_rope_qat(q.view(T, IDX_HEADS, IDX_DIM), positions[:T], self._rope(), impl=impl)
            w = mm_fp32(x[:T], self._w(self.weights_proj, (IDX_HEADS, HIDDEN))) * (
                self.softmax_scale * IDX_HEADS ** -0.5)
        keys_all = self._key_rows()
        keys_of: dict[int, torch.Tensor] = {}
        for kind, r, a, b in tasks:
            if kind == "empty":
                self._write_empty(topk_out, cand_out, a, b)
                continue
            t0, t1, n, key_tile = info[r]
            ends = md.num_visible[t0:t1]
            if kind == "cand" and fused:         # CAND_ALL rows are expanded in-kernel
                with _segment("idx_cand"):
                    topk_out[a:b] = indexer_fused.fused_candidate_topk(
                        q[a:b], w[a:b], cand_out[a:b], ends[a - t0:b - t0], keys_all,
                        md.token_to_req_indices[a:b], md.block_table, md.storage_block_size)
                continue
            if kind == "cand":
                # every token of the request has a proper candidate list: score <= 2048 x 8 keys per token instead
                # of all n (the full scan below is O(n) per token; at n = 1M it dominated prefill and decode)
                c = cand_out[a:b]
                if mode == "auto":                      # CAND_ALL rows: blocks 0..2047 (as the device path)
                    blocks = torch.arange(CAND_TOPK_BLOCKS, device=x.device, dtype=c.dtype)[None, :]
                    c = torch.where(c[:, :1] == CAND_ALL, blocks, c)
                with _segment("idx_cand"):
                    topk_out[a:b] = candidate_topk(
                        q[a:b], w[a:b], c, ends[a - t0:b - t0], keys_all,
                        lambda pos, a=a, b=b: logical_to_rows(pos, md.token_to_req_indices[a:b], md.block_table,
                                                              md.storage_block_size))
                continue
            keys = keys_of.get(r)
            if keys is None:
                with _segment("idx_gather"):
                    j = torch.arange(n, device=x.device, dtype=torch.int64)[None, :]
                    req = md.token_to_req_indices[t0:t0 + 1]
                    rows = logical_to_rows(j, req, md.block_table, md.storage_block_size)[0]
                    keys = keys_of[r] = keys_all.index_select(0, rows)            # [n, 128] fp16
            with _segment("idx_scan"):
                scores = torch.empty((b - a, n), dtype=torch.float32, device=x.device)
                if fused:
                    indexer_fused.fused_index_scores(q[a:b], w[a:b], keys, scores)
                else:
                    index_scores(q[a:b], w[a:b], keys, scores, key_tile=key_tile)
            e = ends[a - t0:b - t0]
            with _segment("idx_select"):
                if self.topo.is_candidate_source:
                    select_candidate_blocks(scores, e, CAND_TOPK_BLOCKS, CAND_BLOCK, out=cand_out[a:b])
                v2 = fused and self.__dict__.get("topk_v2", False)
                if self.topo.uses_candidates:
                    apply_candidate_mask(scores, e, cand_out[a:b], CAND_BLOCK)
                elif not v2:                                  # v2 applies the causal bound inside topk_sel
                    cols = torch.arange(n, device=x.device)
                    scores.masked_fill_(cols[None, :] >= e[:, None], float("-inf"))
            with _segment("idx_topk"):
                if v2:
                    topk_out[a:b] = indexer_fused.fused_topk_sorted_v2(scores, e)
                else:
                    topk_out[a:b] = (indexer_fused.fused_topk_sorted(scores, e) if fused
                                     else topk_sorted(scores, e, IDX_TOPK))
