# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sequence-parallel hyper-connection glue for stage-0 prefill (``VLLM_DS41_SP_HC=1``; default off = today's path).

Today every TP rank all-reduces each sublayer's [T, 5120] fp32 output (40 x 20.8 MB per 1016-token chunk, ~0.56 ms
each at the NVLink floor) and then ALL FOUR ranks run the identical row-local HC glue on all T rows: hc_mixes (an
FP32 GEMM + Sinkhorn), hc_pre, rmsnorm, hc_post -- ~46 ms per chunk (profile 20261025T021427Z), redundantly x4.

Here the stream is sharded by rows across the TP group for the whole stage-0 prefill forward (rank r owns the
contiguous rows [bounds[r], bounds[r + 1]) of the chunk; shards differ by at most one row, no padding):

  per layer   mixes / pre / norm on the SHARD -> x_shard [S, 5120] fp16 -> all-gather -> x [T, 5120] (byte-identical
              on every rank: each row is computed once and copied) -> attention / MoE as today but WITHOUT their
              all-reduce -> the fp32 partial [T, 5120] is reduce-scattered -> this rank's rows of the fp32 sum ->
              hc_post on the shard.  Engram: the row-parallel wkv GEMM runs on all rows, its all-reduce becomes a
              reduce-scatter and the gate updates the shard.
  stage end   the shards are all-gathered back into full (stream, pre) rows for the PP export (and at every
              layer-sliced-prefill slice boundary, so the slice state keeps today's full-row contract).

Traffic per sublayer: reduce-scatter 3/4 x 20.8 MB + all-gather 3/4 x 10.4 MB = 23.4 MB per rank vs 31.2 MB for the
all-reduce; the glue runs on T/4 rows. The fp32 sublayer-output contract (PORT_DESIGN §4) is kept: the reduce-scatter
sums fp32 partials and hc_post consumes fp32.

Numerics vs today (documented order-change sites, not bit-identical in general; gated by oracle KL + self-consistency):
  (1) the 4-way fp32 sum: NCCL reduce-scatter vs NCCL all-reduce may add the partials in a different order;
  (2) hc_mixes' FP32 GEMM runs with M = S instead of M = T (cuBLAS may pick a different reduction schedule);
  everything else (Sinkhorn, hc_pre, rmsnorm, hc_post, the gathered x) is per-row and byte-identical by construction.
  Sliced vs unsliced SP runs ARE bit-identical to each other (same shards, same kernels, gathers are copies).
Scope: stage 0 (pp_is_first), prefill-sized steps (T > 16), not capturing, no DSpark taps / dumps, tp_size > 1.
Decode graphs and stage 1 keep today's path.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from vllm.models.deepseek_v41 import knobs
from vllm.models.deepseek_v41.common.contracts import HC, HIDDEN, STREAM_DTYPE
from vllm.models.deepseek_v41.common.hc import hc_mixes, hc_post, hc_pre, rmsnorm_to_act

SP_HC_ENV = "VLLM_DS41_SP_HC"
SP_MIN_T = 17                      # prefill-sized steps only (decode / spec-verify steps stay on today's path)


def sp_hc_enabled() -> bool:
    return knobs.env_bool(SP_HC_ENV, False)


def shard_bounds(T: int, size: int) -> list[int]:
    """Contiguous row shards of T rows over ``size`` ranks: sizes T // size (+1 for the first T % size ranks)."""
    base, extra = divmod(T, size)
    out = [0]
    for r in range(size):
        out.append(out[-1] + base + (1 if r < extra else 0))
    return out


@dataclass
class SPGroup:
    """The TP group's row-collectives for one forward. ``all_gatherv`` concatenates the ranks' shards along dim 0
    (sizes = the shard sizes); ``reduce_scatterv`` sums the ranks' [T, ...] tensors and returns this rank's rows."""
    rank: int
    size: int
    all_gatherv: Callable[[torch.Tensor, list[int]], torch.Tensor]
    reduce_scatterv: Callable[[torch.Tensor, list[int]], torch.Tensor]

    @classmethod
    def tp(cls) -> SPGroup:
        from vllm.distributed import get_tp_group

        g = get_tp_group()
        return cls(g.rank_in_group, g.world_size,
                   lambda x, sizes: g.all_gatherv(x, dim=0, sizes=sizes),
                   lambda x, sizes: g.reduce_scatterv(x, dim=0, sizes=sizes))


@dataclass
class SPState:
    """A sharded (stream, pre) during a stage-0 prefill forward."""
    group: SPGroup
    T: int
    bounds: list[int]
    stream: torch.Tensor          # [S, 4, 5120] bf16, this rank's rows
    pre: torch.Tensor             # [S, 4] fp32

    @property
    def sizes(self) -> list[int]:
        return [self.bounds[r + 1] - self.bounds[r] for r in range(self.group.size)]

    @property
    def lo(self) -> int:
        return self.bounds[self.group.rank]

    @property
    def hi(self) -> int:
        return self.bounds[self.group.rank + 1]

    def gather_rows(self, x_shard: torch.Tensor) -> torch.Tensor:
        """[S, ...] -> [T, ...]: every rank receives the same bytes (each row computed once by its owner)."""
        return self.group.all_gatherv(x_shard.contiguous(), self.sizes)

    def scatter_sum(self, partial: torch.Tensor) -> torch.Tensor:
        """[T, ...] fp32 partial on every rank -> this rank's rows of the fp32 sum [S, ...]."""
        if partial.shape[0] != self.T:
            raise ValueError(f"SP reduce-scatter: partial has {partial.shape[0]} rows, the chunk has {self.T}")
        return self.group.reduce_scatterv(partial.contiguous(), self.sizes)


def shard(group: SPGroup, stream: torch.Tensor, pre: torch.Tensor) -> SPState:
    """Full (stream, pre) [T, ...] on every rank -> this rank's shard (a copy: the slice state buffers may be
    reused by the caller)."""
    T = stream.shape[0]
    bounds = shard_bounds(T, group.size)
    lo, hi = bounds[group.rank], bounds[group.rank + 1]
    return SPState(group, T, bounds, stream[lo:hi].contiguous(), pre[lo:hi].contiguous())


def unshard(st: SPState) -> tuple[torch.Tensor, torch.Tensor]:
    """Shards -> full (stream [T, 4, 5120], pre [T, 4]) on every rank."""
    return st.gather_rows(st.stream), st.gather_rows(st.pre)


def sp_block(st: SPState, p, attn_partial: Callable[[torch.Tensor], torch.Tensor],
             moe_partial: Callable[[torch.Tensor], torch.Tensor],
             engram_rows: Callable[[torch.Tensor], torch.Tensor] | None = None,
             engram_gate: Callable[[torch.Tensor, torch.Tensor], None] | None = None) -> torch.Tensor:
    """One decoder block on the shard. ``p`` holds the block's HC parameters (hc_attn_fn / hc_attn_scale /
    hc_attn_base / attn_norm_weight and the ffn counterparts). ``attn_partial(x [T, 5120] fp16) -> [T, 5120] fp32``
    is this rank's un-reduced attention output (``moe_partial`` likewise); ``engram_rows(T) -> [T, 25600] fp32``
    this rank's un-reduced wkv output and ``engram_gate(stream_shard, kv_shard)`` the in-place gate. Updates
    ``st.stream`` / ``st.pre`` in place and returns the block's ffn_pre shard."""
    stream = st.stream
    if engram_rows is not None:
        kv = st.scatter_sum(engram_rows(st.T))                                   # [S, 25600] fp32 summed rows
        engram_gate(stream, kv)                                                  # in place on the shard
    attn_pre, attn_post, attn_comb = hc_mixes(stream, p.hc_attn_fn, p.hc_attn_scale, p.hc_attn_base)
    x = rmsnorm_to_act(hc_pre(stream, st.pre), p.attn_norm_weight)
    out = st.scatter_sum(attn_partial(st.gather_rows(x)))
    stream = hc_post(out, stream, attn_post, attn_comb)
    ffn_pre, ffn_post, ffn_comb = hc_mixes(stream, p.hc_ffn_fn, p.hc_ffn_scale, p.hc_ffn_base)
    x = rmsnorm_to_act(hc_pre(stream, attn_pre), p.ffn_norm_weight)
    out = st.scatter_sum(moe_partial(st.gather_rows(x)))
    st.stream = hc_post(out, stream, ffn_post, ffn_comb)
    st.pre = ffn_pre
    return ffn_pre


def reference_block(stream: torch.Tensor, pre: torch.Tensor, p, attn_full, moe_full,
                    engram_full: Callable[[torch.Tensor], torch.Tensor] | None = None,
                    engram_gate=None) -> tuple[torch.Tensor, torch.Tensor]:
    """Today's full-row block math (DeepseekV41DecoderLayer.forward without the modules): the twin ``sp_block`` is
    checked against. ``attn_full(x) -> [T, 5120] fp32`` is the ALL-REDUCED output."""
    if engram_full is not None:
        engram_gate(stream, engram_full(stream.shape[0]))
    attn_pre, attn_post, attn_comb = hc_mixes(stream, p.hc_attn_fn, p.hc_attn_scale, p.hc_attn_base)
    x = rmsnorm_to_act(hc_pre(stream, pre), p.attn_norm_weight)
    stream = hc_post(attn_full(x), stream, attn_post, attn_comb)
    ffn_pre, ffn_post, ffn_comb = hc_mixes(stream, p.hc_ffn_fn, p.hc_ffn_scale, p.hc_ffn_base)
    x = rmsnorm_to_act(hc_pre(stream, attn_pre), p.ffn_norm_weight)
    return hc_post(moe_full(x), stream, ffn_post, ffn_comb), ffn_pre


def check_shapes(stream: torch.Tensor, pre: torch.Tensor) -> None:
    if stream.dim() != 3 or stream.shape[1:] != (HC, HIDDEN) or stream.dtype != STREAM_DTYPE:
        raise ValueError(f"SP: stream must be [T, {HC}, {HIDDEN}] {STREAM_DTYPE}, got {tuple(stream.shape)} {stream.dtype}")
    if pre.shape != (stream.shape[0], HC) or pre.dtype != torch.float32:
        raise ValueError(f"SP: pre must be [T, {HC}] fp32, got {tuple(pre.shape)} {pre.dtype}")


def sizes_of(bounds: Sequence[int]) -> list[int]:
    return [bounds[i + 1] - bounds[i] for i in range(len(bounds) - 1)]
