# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 on SM70: decoder layer, model, causal LM, weight loading, PP payload (PORT_DESIGN §3.2, §3.5,
§3.7; owner L-CORE).

Attention, MoE and Engram come from the other lanes through the §3 signatures. They are imported when the model
is built (not at module import), so the registry can inspect this class before those lanes merge and tests can
substitute plain-torch stand-ins with the same signatures.

Numerics (§4.1): HC stream BF16 with FP32 math (common/hc.py), sublayer inputs FP16, sublayer outputs FP32,
FP16 head with FP32 logits, FP16 GEMMs with FP32 accumulation (reduced-precision reductions disabled).
"""

from __future__ import annotations

import contextlib
import os
import typing
from dataclasses import dataclass, field
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any

import numpy as np
import regex as re
import torch
import torch.nn as nn

from vllm.config import CUDAGraphMode, VllmConfig
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.distributed.utils import get_pp_indices
from vllm.forward_context import get_forward_context as _forward_context   # forward() imports the bare name locally
from vllm.logger import init_logger
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.models.utils import PPMissingLayer, extract_layer_index, make_layers, maybe_prefix
from vllm.sequence import IntermediateTensors
from vllm.models.deepseek_v41.sm70 import dspark_taps, fixed_m

from .. import knobs
from ..common import contracts as C
from ..common import dump as ds41_dump
from ..common.contracts import LayerTopology, SharedAttnBuffers, StagePlan, allocate_shared_attn_buffers
from ..common.hc import hc_collapse, hc_expand, hc_mixes, hc_post, hc_pre, rmsnorm_to_act
from ..common.topology import N_BACKBONE_LAYERS, layer_topology, make_stage_plan
from .replay import ReplayController, RowPlan, replay_rows_k
from . import sp_hc

if TYPE_CHECKING:
    from ..common.engram_host import EngramHostService
    from .moe import ExpertSpillPlan

# Model knobs (PORT_DESIGN §5.5: "--ds41-*" flags are L-CORE knobs)
ENGRAM_DIR_KNOB = "VLLM_DS41_CORE_ENGRAM_DIR"
SPILL_KNOB = "VLLM_DS41_CORE_SPILL_EXPERTS_PER_LAYER"
SPILL_RANKING_KNOB = "VLLM_DS41_CORE_SPILL_RANKING"   # dir of layer<L>.npy routing counts (VLLM_DS41_MOE_ROUTE_HIST)
LAYER_SUBSET_KNOB = "VLLM_DS41_CORE_LAYER_SUBSET"   # e.g. "0,1,2,3": run only these backbone layers (I1 / smoke)
LAYER_SUBSET_ALLOW_KNOB = "VLLM_DS41_CORE_ALLOW_LAYER_SUBSET"   # must be 1 too: a subset is a debug-only partial model
# L-ATTN's cross-stage KV-mirror check (ds41/attn kv_mirror.py MIRROR_CHECK_ENV / PP_KEY_KV20_CRC; contracts.py is
# frozen, so the extra payload key lives here and is cross-checked against kv_mirror when the knob is on)
MIRROR_CHECK_KNOB = "VLLM_DS41_ATTN_MIRROR_CHECK"
KV20_CRC_KEY = "kv20_crc"

logger = init_logger(__name__)
_LAYER_RE = re.compile(r"^layers\.(\d+)\.")
_SKIP_PREFIXES = ("vision.", "aligner.", "image_", "mtp.")


@dataclass
class _LayerDebug:
    """DS41_DEBUG_POS: the traced row, its per-layer norms and the optional DS41_DUMP_LAYERS dump."""
    row: int
    norms: list = field(default_factory=list)
    dump: dict | None = None


# ---------------------------------------------------------------- configuration checks
def check_contract_constants(config: Any) -> None:
    """The frozen contract constants (§3.1) must equal the checkpoint's config."""
    rope_theta = getattr(config, "rope_theta", None)
    if rope_theta is None:
        rope_theta = (getattr(config, "rope_parameters", None) or {}).get("rope_theta")
    pairs = (
        ("hidden_size", C.HIDDEN), ("hc_mult", C.HC), ("num_attention_heads", C.N_HEADS), ("head_dim", C.HEAD_DIM),
        ("qk_rope_head_dim", C.ROPE_DIM), ("q_lora_rank", C.Q_LORA), ("o_groups", C.O_GROUPS),
        ("o_lora_rank", C.O_LORA), ("sliding_window", C.WINDOW), ("index_n_heads", C.IDX_HEADS),
        ("index_head_dim", C.IDX_DIM), ("index_topk", C.IDX_TOPK), ("candidate_source_layer_id", C.CAND_SOURCE),
        ("candidate_block_size", C.CAND_BLOCK), ("candidate_topk_blocks", C.CAND_TOPK_BLOCKS),
        ("engram_head_dim", C.ENGRAM_HEAD_DIM), ("n_routed_experts", C.N_EXPERTS),
        ("num_experts_per_tok", C.TOP_K), ("moe_intermediate_size", C.MOE_INTER),
        ("routed_scaling_factor", C.ROUTED_SCALE), ("swiglu_limit", C.SWIGLU_LIMIT), ("rms_norm_eps", C.NORM_EPS),
        ("hc_eps", C.HC_EPS), ("hc_sinkhorn_iters", C.HC_SINKHORN_ITERS),
        ("compress_rope_theta", C.COMPRESS_ROPE_THETA), ("num_hidden_layers", N_BACKBONE_LAYERS),
    )
    bad = [f"{name}={getattr(config, name, None)!r} (contract {want!r})" for name, want in pairs
           if getattr(config, name, None) != want]
    if rope_theta != C.ROPE_THETA:
        bad.append(f"rope_theta={rope_theta!r} (contract {C.ROPE_THETA!r})")
    engram_subtables = (getattr(config, "engram_max_ngram_size", 0) - 1) * getattr(config, "engram_n_heads", 0)
    if engram_subtables != C.ENGRAM_SUBTABLES:
        bad.append(f"engram subtables {engram_subtables} (contract {C.ENGRAM_SUBTABLES})")
    if bad:
        raise ValueError("DeepSeek-V4.1 config does not match the frozen contracts: " + "; ".join(bad))


def pipeline_partition(num_layers: int, pp_size: int) -> list[int]:
    """Per-stage layer counts exactly as vLLM's make_layers will split them (VLLM_PP_LAYER_PARTITION aware)."""
    counts = []
    for rank in range(pp_size):
        start, end = get_pp_indices(num_layers, rank, pp_size)
        counts.append(end - start)
    return counts


def engram_io_threads(pp_size: int) -> int:
    """Engram pread threads per rank (D11 / AM-12): 2 under PP2, 1 under PP3+ (CPU budget: 12 cores for 8-12
    workers), the service default 4 without PP. VLLM_DS41_ENGRAM_IO_THREADS still overrides inside the service."""
    return 4 if pp_size == 1 else (2 if pp_size == 2 else 1)


def engram_row_dir(model_path: str) -> str:
    """Directory holding the Engram shards (the checkpoint shards that carry ``*.engram.embed.*``).

    VLLM_DS41_CORE_ENGRAM_DIR wins when set (e.g. shards placed on a faster disk); otherwise the model directory
    itself, which is where a plain Hugging Face snapshot keeps them. A model given as a hub id is not a directory
    and needs the knob: fail loud rather than guess."""
    explicit = knobs.env_str(ENGRAM_DIR_KNOB, "")
    if explicit:
        return explicit
    if not os.path.isdir(model_path):
        raise ValueError(
            f"DeepSeek-V4.1 Engram rows are read from the checkpoint shards on disk, but the model {model_path!r} is "
            f"not a local directory; download the checkpoint or set {ENGRAM_DIR_KNOB} to the directory of its shards")
    return model_path


def parse_layer_subset(stage: StagePlan, config: Any) -> frozenset[int] | None:
    """Optional layer subset (knob) restricted to this stage; must be closed under the source relation."""
    raw = knobs.env_str(LAYER_SUBSET_KNOB, "")
    if raw == "":
        return None
    if not knobs.env_bool(LAYER_SUBSET_ALLOW_KNOB, False):
        # MC-CORE F6: a subset inherited from the shell would silently serve a partial model
        raise ValueError(f"{LAYER_SUBSET_KNOB}={raw!r} is set but {LAYER_SUBSET_ALLOW_KNOB} is not 1: a layer subset runs a "
                         "partial (debug) model; set both to use it, or unset the subset")
    try:
        subset = frozenset(int(tok) for tok in raw.split(",") if tok.strip() != "")
    except ValueError as exc:
        raise ValueError(f"{LAYER_SUBSET_KNOB}={raw!r} must be comma-separated layer ids") from exc
    if not subset or any(not 0 <= layer < N_BACKBONE_LAYERS for layer in subset):
        raise ValueError(f"{LAYER_SUBSET_KNOB}={raw!r}: ids must be backbone layers 0..{N_BACKBONE_LAYERS - 1}")
    for layer in sorted(subset):
        topo = layer_topology(config, layer)
        for src in (topo.kv_source, topo.index_source):
            if src is not None and src not in subset:
                raise ValueError(f"{LAYER_SUBSET_KNOB}: layer {layer} needs source layer {src}, which is not in "
                                 f"the subset {sorted(subset)}")
    logger.warning("DeepSeek-V4.1 DEBUG layer subset active: running only backbone layers %s (partial model)",
                   sorted(subset))
    return frozenset(layer for layer in subset if stage.first_layer <= layer <= stage.last_layer)


# ---------------------------------------------------------------- modules
class DeepseekV41Norm(nn.Module):
    """Holds an RMSNorm weight in FP32 (§3.7); the math lives in common.hc.rmsnorm_to_act."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32), requires_grad=False)


def _fp32_param(*shape: int) -> nn.Parameter:
    return nn.Parameter(torch.empty(*shape, dtype=torch.float32), requires_grad=False)


class DeepseekV41DecoderLayer(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str, topo: LayerTopology, stage: StagePlan,
                 shared: SharedAttnBuffers, engram_service: EngramHostService | None,
                 spill: ExpertSpillPlan | None) -> None:
        super().__init__()
        from ..attention import DeepseekV41Attention
        from .moe import DeepseekV41MoE

        self.layer_id = topo.layer_id
        self.topo = topo
        self.engram: nn.Module | None = None
        if topo.has_engram:
            if engram_service is None:
                raise ValueError(f"{prefix}: Engram layer {topo.layer_id} built without an EngramHostService")
            from ..common.engram import DeepseekV41Engram

            self.engram = DeepseekV41Engram(vllm_config, f"{prefix}.engram", topo.layer_id, engram_service)
        self.attn = DeepseekV41Attention(vllm_config, f"{prefix}.attn", topo, stage, shared, aux_streams=None)
        self.ffn = DeepseekV41MoE(vllm_config, f"{prefix}.ffn", topo.layer_id, n_routed_experts=C.N_EXPERTS,
                                  top_k=C.TOP_K, spill=spill)
        self.attn_norm = DeepseekV41Norm(C.HIDDEN)
        self.ffn_norm = DeepseekV41Norm(C.HIDDEN)
        mix, hc_dim = (2 + C.HC) * C.HC, C.HC * C.HIDDEN
        self.hc_attn_fn = _fp32_param(mix, hc_dim)
        self.hc_ffn_fn = _fp32_param(mix, hc_dim)
        self.hc_attn_base = _fp32_param(mix)
        self.hc_ffn_base = _fp32_param(mix)
        self.hc_attn_scale = _fp32_param(3)
        self.hc_ffn_scale = _fp32_param(3)

    def forward(self, stream: torch.Tensor, pre_in: torch.Tensor, positions: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        # stream [T,4,5120] bf16, pre_in [T,4] f32, positions [T] int64  ->  (stream' bf16, ffn_pre f32)
        stream, attn_pre = self.attn_half(stream, pre_in, positions)
        return self.moe_half(stream, attn_pre)

    def attn_half(self, stream: torch.Tensor, pre_in: torch.Tensor, positions: torch.Tensor,
                  part: tuple[int, int] | None = None) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Engram + attention block up to its hc_post -> (stream after attention, attn_pre = this block's pre for the
        FFN collapse). ``part`` = (i, n): attention part i of n (layer-sliced prefill, sm70 attention partial plan);
        parts before the last return None and change nothing but the KV caches / shared top-k rows (recomputing the
        same mixes and projections from the same inputs each part)."""
        dumping = ds41_dump.ENABLED
        layer = self.layer_id
        if dumping:
            ds41_dump.dump(layer, "stream_in", stream)
            ds41_dump.dump(layer, "pre_in", pre_in)
        if self.engram is not None:
            if part is not None:
                raise ValueError(f"layer {layer}: attention parts are not supported on an Engram layer (the Engram "
                                 "update is in place on the stream)")
            stream = self.engram(stream, positions)                    # before the block's mixes (ref:m.py:1262)
            if dumping:
                ds41_dump.dump(layer, "engram.out", stream)
        attn_pre, attn_post, attn_comb = hc_mixes(stream, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        collapsed = hc_pre(stream, pre_in)                                          # uses the PREVIOUS pre
        x = rmsnorm_to_act(collapsed, self.attn_norm.weight)
        if part is None:
            attn_out = self.attn(positions, x)
        else:
            attn_out = self.attn.forward_part(positions, x, part)
            if attn_out is None:
                return None
        if dumping:
            for name, t in (("hc.attn_pre", attn_pre), ("hc.attn_post", attn_post), ("hc.attn_comb", attn_comb),
                            ("hc.attn_x", collapsed), ("attn.x", x), ("attn.out", attn_out)):
                ds41_dump.dump(layer, name, t)
        stream = hc_post(attn_out, stream, attn_post, attn_comb)
        return stream, attn_pre

    def forward_sp(self, st: "sp_hc.SPState", positions: torch.Tensor) -> None:
        """Sequence-parallel HC (VLLM_DS41_SP_HC, sm70/sp_hc.py): the block on this rank's row shard; attention and
        MoE see the all-gathered x of all T rows and return un-reduced partials (reduce=False)."""
        if ds41_dump.ENABLED:
            raise RuntimeError("DS41 dumps are not supported with VLLM_DS41_SP_HC=1")
        eng = self.engram
        sp_hc.sp_block(
            st, _HCParams(self), lambda x: self.attn(positions, x, reduce=False), lambda x: self.ffn(x, reduce=False),
            engram_rows=eng.sp_rows if eng is not None else None, engram_gate=eng.sp_gate if eng is not None else None)

    def moe_half(self, stream: torch.Tensor, attn_pre: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """FFN block: (stream after attention, attn_pre) -> (stream', ffn_pre)."""
        dumping = ds41_dump.ENABLED
        layer = self.layer_id
        ffn_pre, ffn_post, ffn_comb = hc_mixes(stream, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        collapsed = hc_pre(stream, attn_pre)                                        # uses this block's attn pre
        x = rmsnorm_to_act(collapsed, self.ffn_norm.weight)
        if dumping:
            for name, t in (("stream_attn", stream), ("hc.ffn_pre", ffn_pre), ("hc.ffn_post", ffn_post),
                            ("hc.ffn_comb", ffn_comb), ("hc.ffn_x", collapsed), ("moe.x", x)):
                ds41_dump.dump(layer, name, t)
        moe_out = self.ffn(x)
        stream = hc_post(moe_out, stream, ffn_post, ffn_comb)
        if dumping:
            for name, t in (("moe.out", moe_out), ("stream_out", stream), ("pre_out", ffn_pre)):
                ds41_dump.dump(layer, name, t)
        return stream, ffn_pre


class _HCParams:
    """The block's HC parameters as sp_hc.sp_block expects them."""
    __slots__ = ("hc_attn_fn", "hc_attn_scale", "hc_attn_base", "attn_norm_weight", "hc_ffn_fn", "hc_ffn_scale",
                 "hc_ffn_base", "ffn_norm_weight")

    def __init__(self, layer: DeepseekV41DecoderLayer) -> None:
        self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base = layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base
        self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base = layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base
        self.attn_norm_weight, self.ffn_norm_weight = layer.attn_norm.weight, layer.ffn_norm.weight


class DeepseekV41Model(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        check_contract_constants(config)
        if vllm_config.model_config.dtype != torch.float16:
            raise ValueError("DeepSeek-V4.1 on SM70 requires --dtype half (no BF16 tensor cores on Volta); got "
                             f"{vllm_config.model_config.dtype}")
        if vllm_config.parallel_config.use_ubatching:
            # gpu_ubatch_wrapper builds per-ubatch forward contexts without ForwardContext.is_dummy_run, which
            # Engram keys on (PORT_DESIGN §9 AM-2), and the Engram step binding assumes one forward per step.
            raise NotImplementedError("DeepSeek-V4.1 does not support DBO / micro-batching (--enable-dbo, "
                                      "ubatch_size > 1)")
        if ds41_dump.ENABLED and vllm_config.compilation_config.cudagraph_mode != CUDAGraphMode.NONE:
            raise ValueError(f"{ds41_dump.DIR_KNOB} (golden dump) needs eager execution: launch with --enforce-eager")
        # §4.1 MUST: FP16 GEMMs accumulate in FP32 without reduced-precision split-K reductions
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        self.config = config
        self.vllm_config = vllm_config
        pp = get_pp_group()
        self.pp_is_first, self.pp_is_last = pp.is_first_rank, pp.is_last_rank
        self._sp_hc = sp_hc.sp_hc_enabled()   # VLLM_DS41_SP_HC (stage-0 prefill, sm70/sp_hc.py)
        partition = pipeline_partition(config.num_hidden_layers, pp.world_size)
        self.stage = make_stage_plan(config, pp.rank_in_group, pp.world_size, partition)
        self.layer_subset = parse_layer_subset(self.stage, config)
        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        device = torch.get_default_device()
        self.shared = allocate_shared_attn_buffers(max_tokens, self.stage, device)

        self._engram_service: EngramHostService | None = None
        engram_layers = tuple(layer for layer in self.stage.engram_layers if self._in_subset(layer))
        if engram_layers:
            from ..common.engram_host import EngramHostService

            self._engram_service = EngramHostService(
                config, engram_layers, get_tensor_model_parallel_rank(), get_tensor_model_parallel_world_size(),
                engram_row_dir(vllm_config.model_config.model), vllm_config.model_config.tokenizer,
                max_tokens, device, io_threads=engram_io_threads(pp.world_size))
        self.spill: ExpertSpillPlan | None = None
        self.n_spill = knobs.env_int(SPILL_KNOB, 0, minimum=0, maximum=C.N_EXPERTS - 1)
        self.spill_ranking_dir = knobs.env_str(SPILL_RANKING_KNOB, "")
        if self.n_spill > 0:
            from .moe import make_spill_plan

            self.spill = make_spill_plan(self.n_spill, C.N_EXPERTS)

        if self.pp_is_first:
            self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size,
                                                       prefix=f"{prefix}.embed_tokens")
        else:
            self.embed_tokens = PPMissingLayer()

        def build(prefix: str) -> nn.Module:
            layer_id = extract_layer_index(prefix)
            if not self._in_subset(layer_id):
                return PPMissingLayer()
            return DeepseekV41DecoderLayer(vllm_config, prefix, layer_topology(config, layer_id), self.stage,
                                           self.shared, self._engram_service, self._spill_plan(layer_id))

        layers_prefix = f"{prefix}.layers"
        self.start_layer, self.end_layer, self.layers = make_layers(config.num_hidden_layers, build,
                                                                    prefix=layers_prefix)
        if (self.start_layer, self.end_layer) != (self.stage.first_layer, self.stage.last_layer + 1):
            raise RuntimeError(f"make_layers range [{self.start_layer},{self.end_layer}) != stage plan "
                               f"[{self.stage.first_layer},{self.stage.last_layer}]")
        # with VLLM_DS41_ATTN_MIRROR_CHECK=1 the kv-source-20 boundary also carries kv20_crc [T] int32
        self.kv20_crc_check = knobs.env_bool(MIRROR_CHECK_KNOB, False)
        self._kv20_crc: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None
        if self.kv20_crc_check and C.CAND_SOURCE in self.stage.exports_kv_sources + self.stage.mirrored_kv_sources:
            from .. import kv_mirror

            if (kv_mirror.PP_KEY_KV20_CRC, kv_mirror.MIRROR_CHECK_ENV) != (KV20_CRC_KEY, MIRROR_CHECK_KNOB):
                raise RuntimeError(f"kv_mirror's CRC key/knob {kv_mirror.PP_KEY_KV20_CRC!r}/"
                                   f"{kv_mirror.MIRROR_CHECK_ENV!r} != the model's {KV20_CRC_KEY!r}/"
                                   f"{MIRROR_CHECK_KNOB!r}")
            self._kv20_crc = kv_mirror.kv20_crc
        self.mirror: nn.Module | None = None
        if C.CAND_SOURCE in self.stage.mirrored_kv_sources:
            from ..kv_mirror import DeepseekV41KVSourceMirror

            self.mirror = DeepseekV41KVSourceMirror(vllm_config, C.CAND_SOURCE, self.shared)
        self.norm = DeepseekV41Norm(C.HIDDEN) if self.pp_is_last else PPMissingLayer()
        self.replay = self._make_replay(max_tokens, device)

    def _make_replay(self, max_tokens: int, device: torch.device) -> ReplayController | None:
        """VLLM_DS41_REPLAY_ROWS=K > 0 (sm70/replay.py): layers after the candidate source run only each request's
        rows at positions >= N_req - K. Only on the LAST stage that also holds the candidate source (PP2 20/20: stage 1,
        layers 20-39), so the trimmed rows never cross a PP boundary and the logits / DSpark aux are scattered back on
        the stage that consumes them."""
        k = replay_rows_k()
        if k == 0:
            return None
        first = C.CAND_SOURCE + 1
        if not (self.pp_is_last and self.start_layer <= C.CAND_SOURCE and first < self.end_layer
                and self.layer_subset is None):
            logger.info("DeepSeek-V4.1 %s=%d: inactive on this stage (layers %d..%d, last=%s)", "VLLM_DS41_REPLAY_ROWS",
                        k, self.start_layer, self.end_layer - 1, self.pp_is_last)
            return None
        trimmed = [(idx, self.layers[idx]) for idx in range(first, self.end_layer)]
        logger.info("DeepSeek-V4.1 decoder replay: layers %d..%d run the rows at positions >= N_req - %d", first,
                    self.end_layer - 1, k)
        return ReplayController(k, first, trimmed, max_tokens, device)

    def _spill_plan(self, layer_id: int) -> ExpertSpillPlan | None:
        """The layer's expert spill: without SPILL_RANKING the highest ids, else the least-routed experts of the
        layer's decode-step counts (row 1 of ``layer<L>.npy``, written by VLLM_DS41_MOE_ROUTE_HIST)."""
        if self.spill is None or not self.spill_ranking_dir:
            return self.spill
        import numpy as np

        from .moe import spill_plan_from_ranking

        path = os.path.join(self.spill_ranking_dir, f"layer{layer_id:02d}.npy")
        counts = np.load(path)
        if counts.shape != (2, C.N_EXPERTS) or counts[1].sum() <= 0:
            raise ValueError(f"{SPILL_RANKING_KNOB}: {path} has shape {counts.shape} / no decode counts; expected "
                             f"int64 [2, {C.N_EXPERTS}] with row 1 > 0")
        plan = spill_plan_from_ranking(self.n_spill, counts[1])
        spilled_share = counts[1][list(plan.spilled_expert_ids)].sum() / counts[1].sum()
        logger.info("DeepSeek-V4.1 layer %d: ranked spill of %d experts carries %.1f%% of decode routes (%s)",
                    layer_id, self.n_spill, 100.0 * spilled_share, path)
        return plan

    def _in_subset(self, layer_id: int) -> bool:
        return self.layer_subset is None or layer_id in self.layer_subset

    @property
    def engram_service(self) -> EngramHostService | None:
        return self._engram_service

    def owns_layer(self, layer_id: int) -> bool:
        return self.stage.first_layer <= layer_id <= self.stage.last_layer and self._in_subset(layer_id)

    # ---- PP payload (§3.5) ----
    def _schema(self, num_tokens: int, with_kv20: bool) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
        schema = {
            C.PP_KEY_HIDDEN: ((num_tokens, C.HC, C.HIDDEN), C.STREAM_DTYPE),
            C.PP_KEY_PRE_MIX: ((num_tokens, C.HC), C.MIX_DTYPE),
        }
        if with_kv20:
            schema[C.PP_KEY_KV20_CKV] = ((num_tokens, C.CKV_PACKED_DIM), C.CKV_RECORD_DTYPE)
            schema[C.PP_KEY_KV20_IK] = ((num_tokens, C.IK_RECORD_DIM), C.KV_RECORD_DTYPE)
            schema[C.PP_KEY_CAND20] = ((num_tokens, C.CAND_TOPK_BLOCKS), torch.int32)
            if self.kv20_crc_check:
                schema[KV20_CRC_KEY] = ((num_tokens,), torch.int32)
        return schema

    def pp_static_schema(self, num_tokens: int) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
        """Keys/shapes/dtypes of the boundary INTO this stage (receive side), in send order."""
        if self.pp_is_first:
            return {}
        return self._schema(num_tokens, C.CAND_SOURCE in self.stage.mirrored_kv_sources)

    def pp_send_schema(self, num_tokens: int) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
        """Keys/shapes/dtypes this stage sends to the next one (== the next stage's pp_static_schema)."""
        if self.pp_is_last:
            return {}
        return self._schema(num_tokens, C.CAND_SOURCE in self.stage.exports_kv_sources)

    def make_empty_intermediate_tensors(self, batch_size: int, dtype: torch.dtype,
                                        device: torch.device) -> IntermediateTensors:
        # keys for the boundary INTO this stage; ``dtype`` is ignored (the stream is BF16, the mixes FP32)
        schema = self._schema(batch_size, C.CAND_SOURCE in self.stage.mirrored_kv_sources)
        return IntermediateTensors({key: torch.zeros(shape, dtype=dt, device=device)
                                    for key, (shape, dt) in schema.items()})

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    _aux_layers: tuple[int, ...] = ()

    def _set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        """Runner (EAGLE-3) numbering: id L = the stream after layer L - 1. Only this stage's layers can be captured."""
        bad = [l for l in layers if not (self.start_layer < l <= self.end_layer)]
        if bad:
            raise ValueError(f"aux hidden state layers {bad} are not produced on this stage (layers "
                             f"{self.start_layer}..{self.end_layer - 1}); DSpark needs them on the last stage")
        self._aux_layers = tuple(int(l) for l in layers)

    # ---- forward pieces: forward() = _embed_or_ingest -> _run_layers -> _export (or the final collapse);
    # forward_slice() runs a layer range of a staged prefill chunk from persisted state (layer-sliced prefill) ----
    def _embed_or_ingest(self, input_ids: torch.Tensor | None, positions: torch.Tensor,
                         intermediate_tensors: IntermediateTensors | None,
                         inputs_embeds: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """(stream [T,4,5120] bf16, pre [T,4] f32) entering this stage's first layer. Stage 0 embeds + HC-expands;
        later stages read the boundary payload (and ingest the mirrored kv-source-20 records)."""
        if self.pp_is_first:
            embedded = inputs_embeds if inputs_embeds is not None else self.embed_input_ids(input_ids)
            return hc_expand(embedded)
        if intermediate_tensors is None:
            raise RuntimeError("non-first DeepSeek-V4.1 stage called without intermediate tensors")
        stream = intermediate_tensors[C.PP_KEY_HIDDEN]
        pre = intermediate_tensors[C.PP_KEY_PRE_MIX]
        if self.mirror is not None:
            self.mirror.ingest(positions, intermediate_tensors[C.PP_KEY_KV20_CKV],
                               intermediate_tensors[C.PP_KEY_KV20_IK], intermediate_tensors[C.PP_KEY_CAND20],
                               crc=intermediate_tensors[KV20_CRC_KEY] if self.kv20_crc_check else None)
        return stream, pre

    def sp_active(self, T: int, plan: RowPlan | None = None, aux_layers: tuple[int, ...] = (),
                  dbg: _LayerDebug | None = None) -> bool:
        """VLLM_DS41_SP_HC applies to this forward: stage 0, prefill-sized, eager, plain (no replay plan, DSpark taps
        or debug), TP > 1."""
        if not self.__dict__.get("_sp_hc", False) or not self.pp_is_first or T < sp_hc.SP_MIN_T:
            return False
        if plan is not None or aux_layers or dbg is not None or ds41_dump.ENABLED:
            return False
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            return False
        from vllm.distributed import get_tensor_model_parallel_world_size

        return get_tensor_model_parallel_world_size() > 1

    def _run_layers_sp(self, stream: torch.Tensor, pre: torch.Tensor, positions: torch.Tensor, start: int, end: int,
                       group: "sp_hc.SPGroup | None" = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Layers [start, end) with the stream sharded by rows across the TP group (sm70/sp_hc.py); full rows in, full
        rows out (gathered at the end), so callers and the sliced-prefill state keep today's contract."""
        sp_hc.check_shapes(stream, pre)
        st = sp_hc.shard(group if group is not None else sp_hc.SPGroup.tp(), stream, pre)
        for idx in range(start, end):
            layer = self.layers[idx]
            if isinstance(layer, PPMissingLayer):
                continue
            layer.forward_sp(st, positions)
        return sp_hc.unshard(st)

    def _run_layers(self, stream: torch.Tensor, pre: torch.Tensor, positions: torch.Tensor, start: int, end: int,
                    plan: RowPlan | None = None, aux_layers: tuple[int, ...] = (), dbg: _LayerDebug | None = None
                    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, list[torch.Tensor]]:
        """Layers [start, end) -> (stream, pre, positions, kept rows or None, aux). ``plan`` = this forward's
        decoder-replay plan (compacts at replay.first_layer, ``skip`` stops there); ``aux_layers`` = DSpark capture
        ids (runner numbering); ``dbg`` = DS41_DEBUG_POS state."""
        if getattr(self, "_sp_hc", False) and self.sp_active(stream.shape[0], plan, aux_layers, dbg):
            stream, pre = self._run_layers_sp(stream, pre, positions, start, end)
            return stream, pre, positions, None, []
        aux: list[torch.Tensor] = []
        replay = getattr(self, "replay", None)
        full_rows = stream.shape[0]
        kept: torch.Tensor | None = None
        fwd_ctx = md_full = None
        try:
            for idx in range(start, end):
                layer = self.layers[idx]
                if isinstance(layer, PPMissingLayer):   # layer outside the configured subset
                    continue
                if plan is not None and idx == replay.first_layer:
                    fwd_ctx = _forward_context()
                    md_full = fwd_ctx.attn_metadata
                    if plan.kind == "skip":
                        # no row of the step is read past layer 20: zero their window records and stop here
                        replay.zero_dropped(md_full, None)
                        replay.note_skip(plan)
                        break
                    md_kept, rows = replay.compact_step(md_full, plan, self.shared)
                    replay.zero_dropped(md_full, rows.dropped)
                    kept = rows.kept
                    stream, pre, positions = stream.index_select(0, kept), pre.index_select(0, kept), rows.positions
                    fwd_ctx.attn_metadata = md_kept
                stream, pre = layer(stream, pre, positions)
                if dbg is not None and dbg.row >= 0:
                    dbg.norms.append((idx, stream[dbg.row].float().norm()))
                    if dbg.dump is not None:
                        dbg.dump[idx] = stream.to(torch.float16).cpu()
                if idx + 1 in aux_layers:
                    # DSpark target state: the HC stream entering layer idx + 1, averaged over the 4 streams (reference
                    # ``h.mean(dim=2)`` before ``layer(h)``); runner numbering = "output after layer idx" = idx + 1
                    a = dspark_taps.capture_tap(stream)       # VLLM_DS41_DSPARK_TAP_SHIFT: FP16 overflow guard
                    aux.append(a if kept is None else _scatter_rows(a, kept, full_rows))
        finally:
            if fwd_ctx is not None and kept is not None:
                fwd_ctx.attn_metadata = md_full
        return stream, pre, positions, kept, aux

    def _export(self, stream: torch.Tensor, pre: torch.Tensor,
                shared: SharedAttnBuffers | None = None) -> dict[str, torch.Tensor]:
        """The boundary payload this (non-last) stage sends: stream, pre (+ kv-source-20 exports)."""
        num_tokens = stream.shape[0]
        out = {C.PP_KEY_HIDDEN: stream, C.PP_KEY_PRE_MIX: pre}
        if C.CAND_SOURCE in self.stage.exports_kv_sources:
            shared = self.shared if shared is None else shared
            assert shared.export_ckv is not None and shared.export_ik is not None
            out[C.PP_KEY_KV20_CKV] = shared.export_ckv[:num_tokens]
            out[C.PP_KEY_KV20_IK] = shared.export_ik[:num_tokens]
            out[C.PP_KEY_CAND20] = shared.candidate_blocks[:num_tokens]
            if self.kv20_crc_check:
                assert self._kv20_crc is not None
                out[KV20_CRC_KEY] = self._kv20_crc(out[C.PP_KEY_KV20_CKV], out[C.PP_KEY_KV20_IK])
        return out

    def forward(self, input_ids: torch.Tensor | None, positions: torch.Tensor,
                intermediate_tensors: IntermediateTensors | None,
                inputs_embeds: torch.Tensor | None = None) -> torch.Tensor | IntermediateTensors:
        if ds41_dump.ENABLED:
            ds41_dump.begin_step(positions, positions.shape[0])
        stream, pre = self._embed_or_ingest(input_ids, positions, intermediate_tensors, inputs_embeds)
        _dbg_pos = int(__import__("os").environ.get("DS41_DEBUG_POS", "-1"))
        dbg: _LayerDebug | None = None
        if _dbg_pos >= 0 and not torch.cuda.is_current_stream_capturing() and positions.shape[0] <= 2048:
            from vllm.forward_context import get_forward_context, is_forward_context_available
            _dummy = is_forward_context_available() and bool(get_forward_context().is_dummy_run)
            if not _dummy:
                _hit = (positions == _dbg_pos).nonzero()            # one sync, real steps only, never under capture
                if _hit.numel():
                    _row = int(_hit[0, 0])
                    dbg = _LayerDebug(_row, [("in", stream[_row].float().norm())],
                                      {"in": stream.to(torch.float16).cpu()} if __import__('os').environ.get('DS41_DUMP_LAYERS') else None)
        # decoder replay (VLLM_DS41_REPLAY_ROWS): host-only plan from the runner's binding + this step's metadata;
        # None = every layer on every row, exactly today's path (decode / verify steps, graph capture, dummy runs)
        plan: RowPlan | None = None
        replay = getattr(self, "replay", None)
        if replay is not None and not ds41_dump.ENABLED and _dbg_pos < 0:
            plan = replay.begin(_forward_context().attn_metadata)
        full_rows = stream.shape[0]
        stream, pre, positions, kept, aux = self._run_layers(stream, pre, positions, self.start_layer, self.end_layer,
                                                             plan, self._aux_layers, dbg)
        if plan is not None and plan.kind == "skip":
            # rows nobody reads (partial-prefill samples are discarded): a correctly shaped zero hidden / aux
            hidden = torch.zeros((full_rows, C.HIDDEN), dtype=torch.float16, device=stream.device)
            aux.extend(hidden for _ in range(len(self._aux_layers) - len(aux)))
            return (hidden, aux) if self._aux_layers else hidden
        if dbg is not None and dbg.dump is not None and positions.shape[0] > 1:
            dbg.dump["positions"] = positions.cpu()
            import os as _os, torch.distributed as _dist
            torch.save(dbg.dump, f"{_os.environ['DS41_DUMP_LAYERS']}_pp{self.start_layer}_rank{_dist.get_rank()}.pt")
        if dbg is not None:
            logger.info("DS41 layer norms pos %d (row %d of T=%d): %s", _dbg_pos, dbg.row, int(positions.shape[0]),
                        " ".join(f"{k}:{float(v):.3f}" for k, v in dbg.norms))
        if not self.pp_is_last:
            out = self._export(stream, pre)
            if ds41_dump.ENABLED:
                ds41_dump.end_forward()
            return IntermediateTensors(out)
        if not ds41_dump.ENABLED:
            hidden = hc_collapse(stream, pre, self.norm.weight)
            if kept is not None:
                hidden = _scatter_rows(hidden, kept, full_rows)
            return (hidden, aux) if self._aux_layers else hidden
        collapsed = hc_pre(stream, pre)
        hidden = rmsnorm_to_act(collapsed, self.norm.weight, final=True)
        for name, t in (("final.stream_in", stream), ("final.hc", collapsed), ("final.h", hidden)):
            ds41_dump.dump(None, name, t)
        ds41_dump.end_forward()
        return hidden

    # ---- layer-sliced prefill (VLLM_DS41_SLICED_PREFILL; vllm/v1/worker/ds41_sliced_prefill.py) ----
    def forward_slice(self, state: Any, start: int, end: int, part: int = 0, part_index: int = 0,
                      part_count: int = 1) -> None:
        """Run layers [start, end) of the staged chunk held by ``state`` (a SlicedPrefillState) and write the result
        back into it. Eager, called under set_forward_context(state.md_active) by the slice executor, never inside a
        graph capture. Same kernels as an unsliced prefill-only step of the chunk; isolation from the decode step:
        the chunk's own metadata, private SharedAttnBuffers (swapped into the shared object for the slice), no DSpark
        aux capture, the gather-once group cache cleared on entry and exit, Engram rows from the slice lane.

        state: stream/pre/positions (persistent [T_max,...] buffers), rows (current row count: T, or the kept rows
        after a replay compaction), md (chunk metadata) / md_active, shared (private SharedAttnBuffers), needs_init
        + input_ids (stage 0) or handoff (later stages: the boundary dict aliasing the state buffers), n_req,
        replay_done / replay_skip, engram_handle, engram_service.

        Sub-layer slices (ds41_slices.PART_*): ``part`` 1 = attention part ``part_index`` of ``part_count`` of layer
        ``start`` (end == start + 1; the last part leaves (stream after attention, attn_pre) in state.stream /
        state.pre); ``part`` 2 = the MoE half of layer ``start`` from that state, then whole layers start+1..end-1.
        Not on Engram layers; on decoder-replay layers (layer_id >= replay.first_layer) only after the chunk's replay
        plan ran (they then work on the kept rows, as the unsliced forward does)."""
        if not (self.start_layer <= start < end <= self.end_layer):
            raise ValueError(f"forward_slice [{start},{end}) outside this stage's layers [{self.start_layer},"
                             f"{self.end_layer})")
        if part not in (0, 1, 2):
            raise ValueError(f"forward_slice: unknown layer part {part}")
        if part:
            layer = self.layers[start]
            replay = getattr(self, "replay", None)
            if isinstance(layer, PPMissingLayer) or getattr(layer, "engram", None) is not None:
                raise ValueError(f"forward_slice: layer {start} cannot run as sub-layer slices (missing or Engram)")
            if replay is not None and start >= replay.first_layer and not (state.replay_done or state.replay_skip):
                # parts of layers 21-39 run on the chunk's replay-compacted rows / metadata: the replay plan (made in
                # the slice that crosses replay.first_layer) must have run
                raise ValueError(f"forward_slice: layer {start} is a decoder-replay layer and the chunk's replay "
                                 "plan has not run yet")
            if part == 1 and (end != start + 1 or not 0 <= part_index < part_count):
                raise ValueError(f"forward_slice: attention part {part_index}/{part_count} of [{start},{end})")
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("forward_slice must run eagerly (it is called during CUDA graph capture)")
        from .. import attention as _attn_mod
        from . import replay as _replay_mod

        group_rows = getattr(_attn_mod, "_GROUP_ROWS", None)   # gather-once row ids (absent in test stand-ins)

        saved = (self.shared.topk_indices, self.shared.candidate_blocks, self.shared.export_ckv, self.shared.export_ik)
        priv = state.shared
        self.shared.topk_indices, self.shared.candidate_blocks = priv.topk_indices, priv.candidate_blocks
        self.shared.export_ckv, self.shared.export_ik = priv.export_ckv, priv.export_ik
        draft_rows = _replay_mod._LAST_DRAFT_ROWS[0]     # the decode step's DSpark rows: sample_tokens reads them later
        lane = (state.engram_service.slice_lane(state.engram_handle) if state.engram_handle is not None
                else contextlib.nullcontext())
        if group_rows is not None:
            group_rows.clear()
        try:
            with lane:
                self._forward_slice_inner(state, start, end, part, part_index, part_count)
        finally:
            if group_rows is not None:
                group_rows.clear()
            _replay_mod._LAST_DRAFT_ROWS[0] = draft_rows
            (self.shared.topk_indices, self.shared.candidate_blocks, self.shared.export_ckv,
             self.shared.export_ik) = saved

    def _forward_slice_inner(self, state: Any, start: int, end: int, part: int = 0, part_index: int = 0,
                             part_count: int = 1) -> None:
        if state.replay_skip:               # an earlier slice's replay plan dropped every row of layers 21+
            return
        n = state.rows
        positions = state.positions[:n]
        if state.needs_init:
            # first slice of the chunk on this stage: embed (stage 0) / ingest the handoff (later stages)
            stream, pre = self._embed_or_ingest(state.input_ids, positions, state.handoff)
            state.needs_init = False
        else:
            stream, pre = state.stream[:n], state.pre[:n]
        if part == 1:
            # attention part: every part recomputes the block's mixes / projections from the unchanged (stream,
            # pre_in); only the last one produces the post-attention stream (and attn_pre for the MoE half)
            out = self.layers[start].attn_half(stream, pre, positions, part=(part_index, part_count))
            if out is None:
                self._slice_write_back(state, stream, pre, None)       # keep an embedded / ingested input
                return
            self._slice_write_back(state, out[0], out[1], None)
            return
        if part == 2:
            stream, pre = self.layers[start].moe_half(stream, pre)     # state.pre holds the block's attn_pre
            start += 1
            if start == end:
                self._slice_write_back(state, stream, pre, None)
                return
        replay = getattr(self, "replay", None)
        if (replay is not None and not state.replay_done and start <= replay.first_layer < end):
            stream, pre, _, _, _ = self._run_layers(stream, pre, positions, start, replay.first_layer)
            # the same host plan an unsliced prefill-only step of this chunk makes (bind + begin, N_req of the chunk)
            md = state.md_active
            qsl = np.array([0, n], dtype=np.int64)
            replay.bind(1, n, qsl, np.array([state.n_req], dtype=np.int64))
            plan = replay.begin(md)
            state.replay_done = True
            if plan is not None and plan.kind == "skip":
                replay.zero_dropped(md, None)
                state.replay_skip = True
                self._slice_write_back(state, stream, pre, None)
                return
            if plan is not None:
                md_kept, rows = replay.compact_step(md, plan, self.shared)
                replay.zero_dropped(md, rows.dropped)
                stream, pre, positions = stream.index_select(0, rows.kept), pre.index_select(0, rows.kept), rows.positions
                state.md_active = md_kept
                _forward_context().attn_metadata = md_kept
                self._slice_write_back(state, stream, pre, positions)
            start = replay.first_layer
        stream, pre, _, _, _ = self._run_layers(stream, pre, positions, start, end)
        self._slice_write_back(state, stream, pre, None)

    @staticmethod
    def _slice_write_back(state: Any, stream: torch.Tensor, pre: torch.Tensor,
                          positions: torch.Tensor | None) -> None:
        n = stream.shape[0]
        for dst, src in ((state.stream, stream), (state.pre, pre), (state.positions, positions)):
            if src is not None and src.data_ptr() != dst.data_ptr():
                dst[:n].copy_(src)
        state.rows = n

    def slice_handoff(self, state: Any) -> dict[str, torch.Tensor]:
        """The staged chunk's boundary payload to the next stage (non-last stages): stream, pre (+ kv20 exports
        from the chunk's private buffers), all views of slice-owned buffers."""
        n = state.rows
        return self._export(state.stream[:n], state.pre[:n], state.shared)


def _scatter_rows(x: torch.Tensor, rows: torch.Tensor, num_rows: int) -> torch.Tensor:
    """[len(rows), ...] -> [num_rows, ...] with x at ``rows`` and zeros elsewhere (decoder replay: rows past the cut)."""
    out = x.new_zeros((num_rows, *x.shape[1:]))
    return out.index_copy_(0, rows, x)


# ---------------------------------------------------------------- checkpoint names (§3.7)
_STACKED = (
    # (vLLM param fragment, checkpoint fragment, shard id)
    ("attn.fused_wqa_wkv.", "attn.wq_a.", 0),
    ("attn.fused_wqa_wkv.", "attn.wkv.", 1),
    ("compressor.fused_wkv_wgate.", "compressor.wkv.", 0),
    ("compressor.fused_wkv_wgate.", "compressor.wgate.", 1),
    ("shared_experts.gate_up_proj.", "shared_experts.w1.", 0),
    ("shared_experts.gate_up_proj.", "shared_experts.w3.", 1),
)
_STACKED_SHARDS: dict[str, dict[int, str]] = {}   # fused param fragment -> {shard id: checkpoint fragment}
for _param_frag, _ckpt_frag, _shard in _STACKED:
    _STACKED_SHARDS.setdefault(_param_frag, {})[_shard] = _ckpt_frag
_RENAMES = (("shared_experts.w2.", "shared_experts.down_proj."),)
_EXPERT_SCALE_RE = re.compile(r"(\.experts\.\d+\.w[123])\.scale$")
_EXPERT_KEY_RE = re.compile(r"experts\.\d+\.w[123]\.")
_EXL3_SUFFIXES = (".trellis", ".suh", ".svh", ".mul1", ".mcg")


def map_checkpoint_name(name: str) -> str:
    """HF/inference name -> vLLM parameter name (before stacking)."""
    if name == "embed.weight":
        return "model.embed_tokens.weight"
    if name == "head.weight":
        return "lm_head.weight"
    if name == "norm.weight":
        return "model.norm.weight"
    if not _LAYER_RE.match(name):
        raise KeyError(f"unexpected DeepSeek-V4.1 checkpoint tensor {name!r}")
    mapped = "model." + name
    if _EXPERT_SCALE_RE.search(mapped):
        mapped = _EXPERT_SCALE_RE.sub(r"\1.weight_scale", mapped)
    elif mapped.endswith(".scale"):
        mapped = mapped[: -len(".scale")] + ".weight_scale_inv"
    if mapped.endswith(".ffn.gate.bias"):
        mapped = mapped[: -len(".bias")] + ".e_score_correction_bias"
    for old, new in _RENAMES:
        mapped = mapped.replace(old, new)
    return mapped


class DeepseekV41ForCausalLM(nn.Module, SupportsPP):
    supports_eagle3 = True   # aux hidden states for the DSpark drafter (vllm SupportsEagle3 protocol)
    has_own_lm_head = False  # SupportsEagleBase members (the runtime-checkable protocol needs every attribute)
    has_own_embed_tokens = False

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        self.model._set_aux_hidden_state_layers(layers)

    def get_eagle3_default_aux_hidden_state_layers(self) -> tuple[int, ...]:
        # V4.1 reference (inference/model.py:1267-1270): the drafter reads the attention INPUT of layers 37-39, i.e.
        # the stream after layers 36-38 = runner ids 37-39 as given. (V4 captures outputs: the runner adds +1 there;
        # gpu_model_runner._get_eagle3_aux_layers_from_config must not add it for deepseek_v41.)
        ids = getattr(self.config, "dspark_target_layer_ids", None) or ()
        return tuple(int(i) for i in ids)

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        from .moe_exl3 import plan_path
        self._exl3_experts = bool(plan_path())   # routed experts from an EXL3 checkpoint (hybrid model dir)
        self.model = DeepseekV41Model(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))
        if self.model.pp_is_last:
            self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size, prefix=maybe_prefix(prefix, "lm_head"))
            if self.lm_head.num_embeddings_padded != config.vocab_size:
                # compute_logits all-gathers the per-rank shards and trims once: valid only without padding
                raise ValueError(f"LM head pads the vocabulary {config.vocab_size} -> "
                                 f"{self.lm_head.num_embeddings_padded}; compute_logits does not support padding")
        else:
            self.lm_head = PPMissingLayer()
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors  # type: ignore[method-assign]
        self.pp_static_schema = self.model.pp_static_schema
        self.pp_send_schema = self.model.pp_send_schema
        self._delegates: dict[str, nn.Module] = {}

    @property
    def engram_service(self) -> EngramHostService | None:
        return self.model.engram_service

    @property
    def ds41_replay(self) -> ReplayController | None:
        """The decoder-replay controller of this stage (VLLM_DS41_REPLAY_ROWS > 0 on the trimming stage), else None.
        The runner binds each real step's per-request token counts to it right before the forward."""
        return self.model.replay

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(self, input_ids: torch.Tensor | None, positions: torch.Tensor,
                intermediate_tensors: IntermediateTensors | None = None,
                inputs_embeds: torch.Tensor | None = None) -> torch.Tensor | IntermediateTensors:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """[T,5120] fp16 -> [T,129280] f32: FP16 operands, FP32 accumulate AND FP32 output (§4.1)."""
        weight = self.lm_head.weight
        if hidden_states.is_cuda:
            local = fixed_m.mm(hidden_states, weight.t(), out_dtype=torch.float32)
        else:   # CPU (unit tests): same math -- FP16 values are exact in FP32
            local = fixed_m.mm(hidden_states.float(), weight.float().t())
        if get_tensor_model_parallel_world_size() > 1:
            local = tensor_model_parallel_all_gather(local, dim=-1)
        logits = local[:, : self.config.vocab_size]
        if ds41_dump.ENABLED:
            ds41_dump.dump(None, "logits", logits)
            ds41_dump.end_logits()
        return logits

    # ---- loading ----
    def skip_checkpoint_weight(self, name: str) -> bool:
        if name.startswith(_SKIP_PREFIXES) or ".engram.embed." in name or name.endswith(".bias_vl"):
            return True
        if self._exl3_experts:
            # hybrid checkpoint (moe_exl3.py): routed experts from the EXL3 shards, everything else official. vLLM
            # yields every tensor of every listed shard, so drop the official MXFP4 experts and the EXL3 shards'
            # own dense trellis tensors (attention / shared expert / head stay official FP8 here).
            routed = _EXPERT_KEY_RE.search(name) is not None
            if routed and name.endswith((".weight", ".scale")):
                return True
            if not routed and name.endswith(_EXL3_SUFFIXES):
                return True
        match = _LAYER_RE.match(name)
        if match is not None:
            return not self.model.owns_layer(int(match.group(1)))
        if name == "embed.weight":
            return not self.model.pp_is_first
        if name in ("head.weight", "norm.weight"):
            return not self.model.pp_is_last
        return False

    def _delegate_for(self, param_name: str) -> str | None:
        """Prefix of the deepest lane module that loads its own tensors (``load_weights``), e.g. Engram."""
        parts = param_name.split(".")
        for cut in range(len(parts) - 1, 1, -1):
            prefix = ".".join(parts[:cut])
            module = self._delegates.get(prefix)
            if module is not None:
                return prefix
        return None

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params = dict(self.named_parameters())
        self._delegates = {name: module for name, module in self.named_modules()
                           if name.startswith("model.layers.") and name.endswith(".engram")
                           and callable(getattr(module, "load_weights", None))}
        expert_mapping: dict[str, tuple[str, int, str]] = {}
        # N1: a fused / expert-stacked parameter is complete only when EVERY shard reached it -- one shard marks the
        # parameter "loaded" but leaves the other rows torch.empty garbage. parts[param] = the shard keys it got,
        # expected[param] = the full key set (stacked: shard ids; routed experts: (expert_id, shard_id) pairs).
        parts: dict[str, set[object]] = {}
        expected: dict[str, frozenset[object]] = {}
        expert_keys: dict[str, set[object]] = {}
        if any(isinstance(m, DeepseekV41DecoderLayer) for m in self.model.layers):
            from .moe import DeepseekV41MoE

            for param_name, weight_name, expert_id, shard_id in DeepseekV41MoE.expert_params_mapping(C.N_EXPERTS):
                if not _EXPERT_KEY_RE.fullmatch(weight_name):
                    raise ValueError(f"unexpected expert mapping entry {weight_name!r} (expected experts.<e>.w<k>.)")
                expert_mapping[weight_name] = (param_name, expert_id, shard_id)
                expert_keys.setdefault(param_name, set()).add((expert_id, shard_id))
        tp_rank, tp_size = get_tensor_model_parallel_rank(), get_tensor_model_parallel_world_size()
        loaded: set[str] = set()
        delegated: dict[str, list[tuple[str, torch.Tensor]]] = {}
        seen_dense: set[str] = set()
        for ckpt_name, tensor in weights:
            if self.skip_checkpoint_weight(ckpt_name):
                continue
            if self._exl3_experts and ".ffn.experts." not in ckpt_name:
                # the EXL3 shards repeat the official non-expert tensors (norms, router, embeddings, compressors);
                # both copies are the 1017 originals, the first one wins
                if ckpt_name in seen_dense:
                    continue
                seen_dense.add(ckpt_name)
            name = map_checkpoint_name(ckpt_name)
            delegate = self._delegate_for(name)
            if delegate is not None:
                delegated.setdefault(delegate, []).append((ckpt_name.split(".engram.", 1)[1], tensor))
                continue
            if ".ffn.experts." in name:
                mapped, param_frag, key = self._load_expert(params, name, tensor, expert_mapping)
                loaded.add(mapped)
                parts.setdefault(mapped, set()).add(key)
                expected[mapped] = frozenset(expert_keys[param_frag])
                continue
            for param_frag, ckpt_frag, shard_id in _STACKED:
                if ckpt_frag in name:
                    stacked = name.replace(ckpt_frag, param_frag)
                    if stacked in params:   # layer 20's compressor has wkv only (no fused wkv/wgate)
                        param = params[stacked]
                        param.weight_loader(param, tensor, shard_id)
                        loaded.add(stacked)
                        parts.setdefault(stacked, set()).add(shard_id)
                        expected[stacked] = frozenset(_STACKED_SHARDS[param_frag])
                        break
            else:
                if name not in params:
                    raise KeyError(f"checkpoint tensor {ckpt_name!r} -> {name!r} has no parameter in this stage")
                param = params[name]
                loader = getattr(param, "weight_loader", None)
                if loader is None and name.endswith(".attn.attn_sink"):
                    heads = C.N_HEADS // tp_size
                    tensor = tensor.narrow(0, heads * tp_rank, heads)
                (loader or default_weight_loader)(param, tensor)
                loaded.add(name)
        starved = sorted(set(self._delegates) - set(delegated))
        if starved:
            raise RuntimeError(f"no checkpoint tensors reached {starved} (Engram q/k/wkv live in shards 47/48: are "
                               "they present and verified in the model directory?)")
        for prefix, items in delegated.items():
            # PORT_DESIGN §9 AM-1: the module returns the module-relative checkpoint names it consumed
            module = self._delegates[prefix]
            consumed = set(module.load_weights(items))
            given = {rel for rel, _ in items}
            # the delegate may also report the parameters it completed (L-ENGRAM returns 'wkv_r', 'qk' for vLLM's
            # every-parameter-loaded check); those are not checkpoint names. Anything else unknown still raises.
            consumed -= {name for name, _ in module.named_parameters()} - given
            if consumed != given:
                raise RuntimeError(f"{prefix}.load_weights consumed {sorted(consumed)} of {sorted(given)}; "
                                   f"unconsumed {sorted(given - consumed)}, unknown {sorted(consumed - given)}")
            loaded.update(f"{prefix}.{name}" for name, _ in module.named_parameters())
        self._check_complete(loaded, {name: expected[name] - got for name, got in parts.items()})
        return loaded

    @staticmethod
    def _describe_missing_parts(name: str, missing: set[object]) -> str:
        keys = sorted(missing, key=repr)
        if all(isinstance(k, tuple) for k in keys):     # routed experts: (expert_id, shard_id)
            shown = ", ".join(f"expert {e} {shard}" for e, shard in keys[:8])
        else:
            frag = next(f for f in _STACKED_SHARDS if f in name)
            shown = ", ".join(_STACKED_SHARDS[frag][k].rstrip(".") for k in keys[:8])
        return f"{name} lacks {shown}" + (f", ... ({len(keys)} total)" if len(keys) > 8 else "")

    def _check_complete(self, loaded: set[str], missing_parts: dict[str, set[object]] | None = None) -> None:
        """Every parameter that needs checkpoint data must have been filled. vLLM's own completeness check is off
        whenever a quantization method is set (always, for this model), so a missing shard would otherwise leave
        torch.empty garbage / initial values and the server would come up (rule 9). Exempt: zero-size parameters and
        names a module lists in ``ds41_no_checkpoint_params`` (module-relative; parameters it derives itself)."""
        derived: set[str] = set()
        for mod_name, module in self.named_modules():
            for rel in getattr(module, "ds41_no_checkpoint_params", ()):
                derived.add(f"{mod_name}.{rel}" if mod_name else rel)
        missing = sorted(name for name, param in self.named_parameters()
                         if name not in loaded and name not in derived and param.numel() > 0)
        partial = sorted((name, keys) for name, keys in (missing_parts or {}).items() if keys)
        problems: list[str] = []
        if missing:
            shown = ", ".join(missing[:20]) + (f", ... ({len(missing)} total)" if len(missing) > 20 else "")
            problems.append(f"did not provide {len(missing)} parameter(s) of this stage: {shown}")
        if partial:
            shown = "; ".join(self._describe_missing_parts(name, keys) for name, keys in partial[:20])
            more = f"; ... ({len(partial)} total)" if len(partial) > 20 else ""
            problems.append(f"filled only part of {len(partial)} fused/expert parameter(s): {shown}{more}")
        if problems:
            raise RuntimeError(f"DeepSeek-V4.1 checkpoint {' and '.join(problems)}. Check that all 48 shards are "
                               "present and verified in the model directory.")

    @staticmethod
    def _load_expert(params: dict[str, nn.Parameter], name: str, tensor: torch.Tensor,
                     expert_mapping: dict[str, tuple[str, int, str]]) -> tuple[str, str, tuple[int, str]]:
        """Load one routed-expert tensor; returns (parameter name, mapping param fragment, (expert_id, shard_id))."""
        if name.endswith(".weight_scale") and tensor.dtype == torch.float8_e8m0fnu:
            tensor = tensor.view(torch.uint8)   # keep raw E8M0 bytes (copy_ would convert values)
        match = _EXPERT_KEY_RE.search(name)
        entry = expert_mapping.get(match.group(0)) if match is not None else None
        if entry is None:
            raise KeyError(f"routed-expert tensor {name!r} has no entry in DeepseekV41MoE.expert_params_mapping")
        param_name, expert_id, shard_id = entry
        mapped = name.replace(match.group(0), param_name)
        if mapped not in params:
            raise KeyError(f"routed-expert tensor {name!r} -> {mapped!r} has no parameter in this stage")
        param = params[mapped]
        loader = typing.cast(Callable[..., bool], param.weight_loader)
        if not loader(param, tensor, mapped, shard_id=shard_id, expert_id=expert_id, return_success=True):
            raise RuntimeError(f"expert weight_loader refused {name!r} -> {mapped!r} (expert {expert_id})")
        return mapped, param_name, (expert_id, shard_id)
