# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler half of DeepSeek-V4.1 layer-sliced prefill (``VLLM_DS41_SLICED_PREFILL=1``; hol-design.md §2(a), Appendix A).

While at least one request decodes, the scheduler does not run a long prompt's non-final chunks on the normal path
(where a 1024-token chunk ahead of the decoder's step costs the decoder ~0.3-0.5 s). It *stages* the chunk instead:

* staging allocates the chunk's KV blocks (zeroed / demand-mapped by the workers through the step's
  ``new_block_ids_to_zero`` like any allocation), does NOT hash them (``delay_cache_blocks``) and does NOT advance
  ``num_computed_tokens``; the request is not schedulable on the normal path while any of its chunks is in flight;
* every step that carries a decode row then carries ``SliceDirective`` s: stage 0 runs ``VLLM_DS41_SLICE_LAYERS_S0``
  layers of its chunk per step, stage 1 its slices (default: first layer alone, then the rest); stage 0's last slice
  and stage 1's first slice of a chunk are emitted in the same step (the handoff isend/irecv pair), so stage 0 may
  slice chunk k+1 while stage 1 finishes chunk k; at most one chunk per stage;
* the last stage reports ``ds41_slice_done(req_id, chunk_id)``; only then ``num_computed_tokens`` advances and the
  chunk's blocks are hashed into the prefix cache;
* the prompt's final chunk (at least ``MIN_TAIL`` tokens, the last staged chunk is shortened to leave them) always
  runs on the normal path: sampling, the DSpark drafter window and the replay tail are unchanged;
* no decoder left -> chunks still on stage 0/1 are aborted and the prompt continues on today's path (pipelined
  chunked prefill); a chunk whose stage-1 last slice is already submitted completes normally;
* preemption / abort / finish of a staged request -> ``ds41_slice_aborts`` for its in-flight chunks; prompt-logprob,
  structured-output, multimodal, pooling and LoRA requests, preempted (resumed) requests and KV-connector setups
  always use the normal path.

Requests staged straight from the waiting queue have never been sent to the workers: their first normal step sends
them as ``NewRequestData`` with the full block table and ``num_computed_tokens`` = the staged prefix. Requests the
workers already hold get the blocks allocated at staging appended to their ``new_block_ids`` on their next normal
step (``pp_sent_upto`` is untouched by staging, so the non-last stages receive the staged token ids then too).

The planner holds only bookkeeping; ``Scheduler`` calls it at fixed points of ``schedule()`` / ``update_from_output``
/ ``_preempt_request`` / ``_free_request`` / ``update_draft_token_ids``.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import math

from vllm.v1 import ds41_slices
from vllm.v1.ds41_slices import (PART_ATTN, PART_MOE, PART_WHOLE, SliceDirective, plan_layer_slices,
                                 slice_layers_s0, slice_layers_s1)

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
    from vllm.v1.request import Request

# The prompt's last MIN_TAIL tokens always take the normal path (hol-design §2(a) step 6).
MIN_TAIL = 128


def is_decoding(request: Any) -> bool:
    """Same predicate as the decode lane: past the prompt at schedule-time accounting."""
    return request.num_computed_tokens >= request.num_prompt_tokens


@dataclass
class _Chunk:
    req_id: str
    chunk_id: int
    start: int
    num_tokens: int
    token_ids: tuple[int, ...]
    n_req: int
    layer: int = -1            # next layer on the stage that currently owns the chunk (absolute id)
    part: int = 0              # stage 1 sub-layer slices: next attention part of the current (heavy) layer
    parts: int = -1            # attention parts of the current layer (-1 = not decided yet, 0 = no split)
    attn_done: bool = False    # every attention part ran: the layer's MoE part is next
    s1_started: bool = False   # a stage-1 directive of the chunk was emitted

    @property
    def end(self) -> int:
        return self.start + self.num_tokens

    @property
    def key(self) -> tuple[str, int]:
        return (self.req_id, self.chunk_id)


@dataclass
class SliceStats:
    staged: int = 0
    done: int = 0
    aborted: int = 0
    slices: int = 0
    stalls: int = 0
    by_reason: dict[str, int] = field(default_factory=dict)


class _Ema:
    def __init__(self, value: float, alpha: float):
        self.value, self.alpha, self.n = float(value), alpha, 0

    def add(self, x: float) -> None:
        self.n += 1
        self.value = x if self.n == 1 else self.value + self.alpha * (x - self.value)


class AdaptiveSlicePolicy:
    """VLLM_DS41_SLICE_POLICY=adaptive: size every slice from measured costs (hol-design Phase 2).

    Per stage s, from the workers' feedback (ModelRunnerOutput.ds41_slice_ms / ds41_decode_ms):
      per_layer[s]  EMA of ms/layer of non-first slices;   head[s]  EMA of a first slice's extra cost (chunk setup,
                    metadata build, Engram begin on stage 0; the non-replayed layer 20 on stage 1)
      fwd[s]        EMA of the stage's decode forward (last stage: + sampling/drafter post)
      span          EMA of the last stage's decode span on CLEAN steps (no slices in this or the previous step), i.e.
                    the undisturbed decode cycle the slices must hide in.
    Budget of stage s in a step = span / keep - fwd[s] - margin (keep = 1: slices only fill the idle gap; keep < 1:
    the decode cycle may stretch to span / keep). Layers x = budget / per_layer (after head on a first slice); the
    stage's remaining layers are split into round(remaining / x) near-equal slices, fractions carried across steps,
    clamped to [1, remaining].
    Until a stage has samples the priors below (Appendix A numbers) apply."""

    PRIORS = {"span": 70.0, "fwd": (32.6, 36.8), "per_layer": (14.8, 11.4), "head": (0.0, 0.0)}

    def __init__(self, keep: float | None = None, margin_ms: float | None = None, alpha: float = 0.25):
        self.keep = ds41_slices.slice_decode_keep() if keep is None else keep
        self.margin = ds41_slices.slice_margin_ms() if margin_ms is None else margin_ms
        p = self.PRIORS
        self.span = _Ema(p["span"], alpha)
        self.fwd = [_Ema(v, alpha) for v in p["fwd"]]
        self.per_layer = [_Ema(v, alpha) for v in p["per_layer"]]
        self.head = [_Ema(v, alpha) for v in p["head"]]
        self.carry = [0.0, 0.0]
        # stage-1 sub-layer slices: attention half of the first layer = attn_fixed + scan_b * T * n_ctx / 1e6
        self.attn_fixed = ds41_slices.slice_s1_attn_fixed_ms()
        self.scan_b = _Ema(ds41_slices.slice_s1_scan_ms_per_mtok2(), alpha)
        # stage-1 per-layer-type costs on the rows a layer runs (per 1024 rows): plain layer, reindex attention half
        self.plain = _Ema(ds41_slices.slice_s1_layer_ms(), alpha)
        # reindex attention: measured / prior ratio (the prior, ds41_slices.reindex_attn_prior_ms, already follows the
        # scoring mode and n; the samples carry it as their work units)
        self.reindex_ratio = _Ema(1.0, alpha)

    def observe(self, slice_ms=(), decode_ms=()) -> None:
        for sample in slice_ms:
            stage, layers, ms, first = sample[:4]
            if int(layers) < 0:          # one layer's attention half (sum of its parts): (stage, -1, ms, F, work, kind)
                work = float(sample[4]) if len(sample) > 4 else 0.0
                kind = int(sample[5]) if len(sample) > 5 else ds41_slices.COST_KIND_SCAN
                if work > 0 and float(ms) > 0:
                    if kind == ds41_slices.COST_KIND_REINDEX_ATTN:
                        if work >= 0.5:                 # prior ms of the sample (rows / mode / n aware)
                            self.reindex_ratio.add(float(ms) / work)
                    else:
                        self.scan_b.add(max(0.0, float(ms) - self.attn_fixed) / work)
                continue
            if len(sample) > 6 and int(sample[5]) == ds41_slices.COST_KIND_LAYERS:
                work, n_reindex = float(sample[4]), int(sample[6])
                if n_reindex == 0 and work >= 0.1 and float(ms) > 0:      # plain layers only (skips have work 0)
                    self.plain.add(float(ms) / work)
            stage, layers, ms = int(stage), max(int(layers), 1), float(ms)
            if stage not in (0, 1) or ms <= 0:
                continue
            if first:
                self.head[stage].add(max(0.0, ms - layers * self.per_layer[stage].value))
            else:
                self.per_layer[stage].add(ms / layers)
        for stage, fwd, span, clean in decode_ms:
            stage = int(stage)
            if stage not in (0, 1):
                continue
            if fwd > 0:
                self.fwd[stage].add(float(fwd))
            if stage == 1 and clean and span > 0:
                self.span.add(float(span))

    def budget(self, stage: int) -> float:
        return self.span.value / self.keep - self.fwd[stage].value - self.margin

    def cost(self, stage: int, layers: int, first: bool) -> float:
        return layers * self.per_layer[stage].value + (self.head[stage].value if first else 0.0)

    def choose(self, stage: int, remaining: int, first: bool, budget: float) -> tuple[int, float]:
        """(layers, carry after taking them) for a slice with ``budget`` ms left on this stage this step.

        x = layers the budget affords; the stage's remaining layers are split into k = round(remaining / x) near-equal
        slices (no 1-layer leftover at the end of a chunk), fractional sizes carried across steps."""
        pl = max(self.per_layer[stage].value, 1e-3)
        x = max(0.0, budget - (self.head[stage].value if first else 0.0)) / pl
        if x < 1.0:
            return 1, 0.0
        k = max(1, round(remaining / x))
        if k == 1:
            return remaining, 0.0
        per = remaining / k + self.carry[stage]
        n = min(max(int(per), 1), remaining)
        return n, min(max(per - n, 0.0), 0.999)


class SlicePlanner:
    """Staging + per-step directive emission for one engine (PP2)."""

    def __init__(self, s0_layers: tuple[int, int], s1_layers: tuple[int, int], chunk_tokens: int,
                 s0_per_slice: int | None = None, s1_per_slice: int | None = None, min_tail: int = MIN_TAIL,
                 policy: AdaptiveSlicePolicy | None = None, s1_attn_parts: int | str | None = None):
        if chunk_tokens <= min_tail:
            raise ValueError(f"chunk size {chunk_tokens} must exceed the normal-path tail {min_tail}")
        self.s0_layers, self.s1_layers = s0_layers, s1_layers
        self.s0_ranges = plan_layer_slices(*s0_layers, slice_layers_s0() if s0_per_slice is None else s0_per_slice)
        self.s1_ranges = plan_layer_slices(*s1_layers, slice_layers_s1() if s1_per_slice is None else s1_per_slice)
        self.fixed_next = ({a: b for a, b in self.s0_ranges}, {a: b for a, b in self.s1_ranges})
        self.policy = policy                                  # None = fixed ranges (today)
        self._budget = [0.0, 0.0]
        # stage 1 sub-layer slices of its first layer (VLLM_DS41_SLICE_S1_ATTN_PARTS; 0 = off, today)
        self.s1_attn_parts = ds41_slices.slice_s1_attn_parts() if s1_attn_parts is None else s1_attn_parts
        self.s1_part_ms = ds41_slices.slice_s1_part_ms()
        self.qtile_cap = ds41_slices.split_qtile_cap() or ds41_slices.DEFAULT_SPLIT_QTILE_MAX
        self.replay_k = ds41_slices.replay_rows_k()
        self.replay_first = ds41_slices.replay_first_layer()
        self.reindex = frozenset(ds41_slices.reindex_layers())
        self.max_parts = 8
        self.cand_tile = 16                 # candidate-scoring query-row tile (indexer.candidate_tile at 128 MiB)
        self.chunk_tokens = chunk_tokens
        self.min_tail = min_tail
        self.s0: _Chunk | None = None                         # chunk on stage 0
        self.s1: _Chunk | None = None                         # chunk on stage 1
        self.awaiting: dict[tuple[str, int], _Chunk] = {}     # stage-1 last slice submitted, slice_done pending
        self.in_flight: dict[str, deque[_Chunk]] = {}         # per request, oldest first
        self.next_chunk_id: dict[str, int] = {}
        self.unseen: set[str] = set()                         # staged from waiting, never sent to the workers
        self.pending_blocks: dict[str, KVCacheBlocks] = {}    # seen requests: blocks allocated at staging
        self.pending_aborts: list[tuple[str, int]] = []
        self.normal_ok: set[str] = set()                      # fresh prompts whose rest is final after prefix hits
        self.stats = SliceStats()

    # ------------------------------------------------------------------ queries
    def is_staged(self, req_id: str) -> bool:
        return bool(self.in_flight.get(req_id))

    def staged_upto(self, request: Request) -> int:
        q = self.in_flight.get(request.request_id)
        return q[-1].end if q else request.num_computed_tokens

    @staticmethod
    def eligible(request: Request) -> bool:
        """Requests that may be staged at all (everything else always takes the normal path)."""
        sp = request.sampling_params
        if sp is None or request.pooling_params is not None:
            return False
        if sp.prompt_logprobs is not None or request.use_structured_output:
            return False
        if request.has_encoder_inputs or request.lora_request is not None:
            return False
        if request.prompt_embeds is not None:
            return False
        return True

    def wants_stage(self, request: Request, computed: int | None = None) -> bool:
        """Eligible and the next chunk from ``computed`` (default: num_computed_tokens) would not be the final one."""
        if not self.eligible(request):
            return False
        c = request.num_computed_tokens if computed is None else computed
        return request.num_tokens - c > self.chunk_tokens

    def blocks_normal(self, request: Request, decoders: bool) -> bool:
        """The running loop must not schedule this request: a chunk of it is in flight, or a decoder exists and its
        next chunk is non-final (it waits for stage 0 instead of stalling the decoder)."""
        if self.is_staged(request.request_id):
            return True
        return decoders and self.wants_stage(request)

    def defer_waiting(self, request: Request, decoders: bool) -> bool:
        """The waiting loop must not admit this request on the normal path (fresh long prompt while decoding)."""
        from vllm.v1.request import RequestStatus

        return (decoders and request.status == RequestStatus.WAITING and request.num_computed_tokens == 0
                and request.request_id not in self.normal_ok and self.wants_stage(request))

    # ------------------------------------------------------------------ staging
    def chunk_len(self, request: Request, start: int) -> int:
        """Tokens of the chunk staged at ``start``, or 0 when the rest must take the normal path."""
        remaining = request.num_tokens - start
        if remaining <= self.chunk_tokens:
            return 0
        return min(self.chunk_tokens, remaining - self.min_tail)

    def can_stage(self) -> bool:
        return self.s0 is None

    def add_chunk(self, request: Request, start: int, num_tokens: int, *, unseen: bool,
                  new_blocks: KVCacheBlocks | None) -> _Chunk:
        rid = request.request_id
        cid = self.next_chunk_id.get(rid, 0)
        self.next_chunk_id[rid] = cid + 1
        chunk = _Chunk(rid, cid, start, num_tokens, tuple(request.all_token_ids[start:start + num_tokens]),
                       request.num_tokens)
        self.in_flight.setdefault(rid, deque()).append(chunk)
        self.s0 = chunk
        if unseen:
            self.unseen.add(rid)
        elif new_blocks is not None:
            prev = self.pending_blocks.get(rid)
            self.pending_blocks[rid] = new_blocks if prev is None else prev + new_blocks
        self.stats.staged += 1
        return chunk

    # ------------------------------------------------------------------ per-step emission
    def emit(self, kv_cache_manager: KVCacheManager, *, decode_in_step: bool) -> tuple[SliceDirective, ...]:
        """Directives for this step. Only steps that carry a decode row carry slices (the workers run them after the
        decode batch); otherwise every chunk waits where it is."""
        if not decode_in_step or (self.s0 is None and self.s1 is None):
            if self.s0 is not None or self.s1 is not None:
                self.stats.stalls += 1
            return ()
        if self.policy is not None:
            self._budget = [self.policy.budget(0), self.policy.budget(1)]
        out: list[SliceDirective] = []
        if self.s1 is not None:
            out.append(self._advance(kv_cache_manager, self.s1, 1))
        if self.s0 is not None:
            chunk = self.s0
            a, b, carry = self._plan(chunk, 0)
            is_last = b == self.s0_layers[1]
            if is_last and self.s1 is not None:
                self.stats.stalls += 1            # stage 1 still busy: hold the handoff (never with the defaults)
            else:
                out.append(self._commit(kv_cache_manager, chunk, 0, a, b, carry))
                if is_last:
                    self.s0 = None
                    chunk.layer = self.s1_layers[0]
                    self.s1 = chunk
                    out.append(self._advance(kv_cache_manager, chunk, 1))
        self.stats.slices += len(out)
        return tuple(out)

    def _plan(self, chunk: _Chunk, stage: int) -> tuple[int, int, float]:
        """Next slice [a, b) of ``chunk`` on ``stage`` (not committed) and the policy carry after it."""
        lo, hi = self.s0_layers if stage == 0 else self.s1_layers
        a = lo if chunk.layer < 0 else chunk.layer
        if self.policy is None:
            return a, self.fixed_next[stage][a], 0.0
        n, carry = self.policy.choose(stage, hi - a, a == lo, self._budget[stage])
        return a, a + n, carry

    # ---- stage-1 cost model (sub-layer planner) ----
    def _s1_rows(self, chunk: _Chunk, layer: int) -> int:
        """Rows ``layer`` runs for ``chunk``: all of them up to the candidate source; past it (decoder replay) only
        positions >= n_req - K (the replay plan's keep rule; K = 0: all rows)."""
        if layer < self.replay_first or self.replay_k <= 0:
            return chunk.num_tokens
        return max(0, chunk.end - max(chunk.start, chunk.n_req - self.replay_k))

    def _coef(self):
        """(layer-20 attention fixed ms, its scan ms per 1e6 T*n, plain ms per 1024 rows, reindex measured/prior)."""
        p = self.policy
        if p is not None:
            return p.attn_fixed, p.scan_b.value, p.plain.value, p.reindex_ratio.value
        return (ds41_slices.slice_s1_attn_fixed_ms(), ds41_slices.slice_s1_scan_ms_per_mtok2(),
                ds41_slices.slice_s1_layer_ms(), 1.0)

    def _splittable(self, chunk: _Chunk, layer: int) -> bool:
        """Layers that run as attention parts: stage 1's first layer (full-context scan) and the reindex layers
        (candidate scoring) when they have rows to run."""
        return layer == self.s1_layers[0] or (layer in self.reindex and self._s1_rows(chunk, layer) > 0)

    def s1_attn_ms(self, chunk: _Chunk, layer: int) -> float:
        fixed, scan_b, plain, reindex = self._coef()
        rows = self._s1_rows(chunk, layer)
        if layer == self.s1_layers[0]:
            return fixed + scan_b * chunk.num_tokens * chunk.end / 1e6
        if layer in self.reindex:
            return reindex * ds41_slices.reindex_attn_prior_ms(rows, chunk.end)
        return (1.0 - ds41_slices.MOE_SHARE) * plain * rows / 1024.0

    def s1_moe_ms(self, chunk: _Chunk, layer: int) -> float:
        return ds41_slices.MOE_SHARE * self._coef()[2] * self._s1_rows(chunk, layer) / 1024.0

    def s1_layer_ms(self, chunk: _Chunk, layer: int) -> float:
        return self.s1_attn_ms(chunk, layer) + self.s1_moe_ms(chunk, layer)

    def _s1_budget(self) -> float:
        return max(self.policy.budget(1), 5.0) if self.policy is not None else self.s1_part_ms

    def attn_parts_for(self, chunk: _Chunk, layer: int | None = None) -> int:
        """Attention parts of ``layer`` (default: stage 1's first layer) for ``chunk``; 0 = run it whole (it fits
        one slice, or it is not splittable). Fixed N or auto = ceil(attention estimate / budget), capped by the
        layer's query-row tiles (scan: VLLM_DS41_INDEXER_QTILE_MAX; candidate scoring: 16-row tiles)."""
        layer = self.s1_layers[0] if layer is None else layer
        if not self._splittable(chunk, layer):
            return 0
        rows = self._s1_rows(chunk, layer)
        tile = self.qtile_cap if layer == self.s1_layers[0] else self.cand_tile
        tiles = max(1, min(self.max_parts, math.ceil(rows / max(tile, 1))))
        mode = self.s1_attn_parts
        if mode != "auto":
            return min(int(mode), tiles) if int(mode) > 0 else 0
        budget = self._s1_budget()
        if self.s1_layer_ms(chunk, layer) <= budget:
            return 0
        return min(max(1, math.ceil(self.s1_attn_ms(chunk, layer) / budget)), tiles)

    def _advance_s1(self, kv_cache_manager: KVCacheManager, chunk: _Chunk) -> SliceDirective:
        """Stage 1 with the sub-layer planner: one directive per step. Heavy layers (first layer, reindex layers with
        kept rows) as attention parts sized to the budget, then a MoE part; whole layers packed up to the budget, at
        most one reindex layer per slice, never fewer than one unit."""
        lo, hi = self.s1_layers
        budget = self._s1_budget() if self.policy is None else max(self._budget[1], 5.0)
        L = lo if chunk.layer < 0 else chunk.layer
        first = not chunk.s1_started
        chunk.s1_started = True
        if not chunk.attn_done and self._splittable(chunk, L):
            if chunk.parts < 0:
                chunk.parts = self.attn_parts_for(chunk, L)
            if chunk.parts > 0:
                i, n = chunk.part, chunk.parts
                chunk.part += 1
                chunk.attn_done = chunk.part == n
                self._spend(self.s1_attn_ms(chunk, L) / n)
                return self._directive(kv_cache_manager, chunk, 1, L, L + 1, first=first, last=False,
                                       layer_part=PART_ATTN, part_index=i, part_count=n, layer_after=L)
        part = PART_MOE if chunk.attn_done else PART_WHOLE
        total = self.s1_moe_ms(chunk, L) if chunk.attn_done else self.s1_layer_ms(chunk, L)
        n_reindex = 1 if (not chunk.attn_done and L in self.reindex and self._s1_rows(chunk, L) > 0) else 0
        b = L + 1
        while b < hi:
            c = self.s1_layer_ms(chunk, b)
            heavy_reindex = b in self.reindex and self._s1_rows(chunk, b) > 0
            if total + c > budget or (heavy_reindex and n_reindex >= 1):
                break
            if self._splittable(chunk, b) and self.attn_parts_for(chunk, b) > 0:
                break                        # it needs its own attention parts: start them next step
            total += c
            n_reindex += int(heavy_reindex)
            b += 1
        chunk.part, chunk.parts, chunk.attn_done = 0, -1, False
        self._spend(total)
        d = self._directive(kv_cache_manager, chunk, 1, L, b, first=first, last=b == hi, layer_part=part,
                            layer_after=b)
        if d.last:
            self.s1 = None
            self.awaiting[chunk.key] = chunk
        return d

    def _spend(self, ms: float) -> None:
        if self.policy is not None:
            self._budget[1] -= ms

    def _advance(self, kv_cache_manager: KVCacheManager, chunk: _Chunk, stage: int) -> SliceDirective:
        if stage == 1 and self.s1_attn_parts:
            return self._advance_s1(kv_cache_manager, chunk)
        a, b, carry = self._plan(chunk, stage)
        d = self._commit(kv_cache_manager, chunk, stage, a, b, carry)
        if stage == 1 and d.last:
            self.s1 = None
            self.awaiting[chunk.key] = chunk
        return d

    def _directive(self, kv_cache_manager: KVCacheManager, chunk: _Chunk, stage: int, a: int, b: int, *, first: bool,
                   last: bool, layer_part: int = PART_WHOLE, part_index: int = 0, part_count: int = 1,
                   layer_after: int) -> SliceDirective:
        chunk.layer = layer_after
        block_ids = kv_cache_manager.get_block_ids_for_computed_tokens(chunk.req_id, chunk.end)
        return SliceDirective(
            stage=stage, req_id=chunk.req_id, chunk_id=chunk.chunk_id, start_pos=chunk.start,
            num_tokens=chunk.num_tokens, token_ids=chunk.token_ids if stage == 0 else (),
            block_ids=tuple(tuple(g) for g in block_ids), layer_start=a, layer_end=b, first=first, last=last,
            n_req=chunk.n_req, layer_part=layer_part, part_index=part_index, part_count=part_count)

    def _commit(self, kv_cache_manager: KVCacheManager, chunk: _Chunk, stage: int, a: int, b: int,
                carry: float) -> SliceDirective:
        lo, hi = self.s0_layers if stage == 0 else self.s1_layers
        if self.policy is not None:
            self.policy.carry[stage] = carry
            self._budget[stage] -= self.policy.cost(stage, b - a, a == lo)
        chunk.layer = b
        block_ids = kv_cache_manager.get_block_ids_for_computed_tokens(chunk.req_id, chunk.end)
        return SliceDirective(
            stage=stage,
            req_id=chunk.req_id,
            chunk_id=chunk.chunk_id,
            start_pos=chunk.start,
            num_tokens=chunk.num_tokens,
            token_ids=chunk.token_ids if stage == 0 else (),
            block_ids=tuple(tuple(g) for g in block_ids),
            layer_start=a,
            layer_end=b,
            first=a == lo,
            last=b == hi,
            n_req=chunk.n_req,
        )

    def observe(self, slice_ms=(), decode_ms=()) -> None:
        """Worker cost feedback (adaptive policy; ignored when fixed)."""
        if self.policy is not None and (slice_ms or decode_ms):
            self.policy.observe(slice_ms, decode_ms)

    # ------------------------------------------------------------------ completion / aborts
    def complete(self, key: tuple[str, int]) -> _Chunk | None:
        """slice_done from the last stage. Returns the chunk to commit, or None for an aborted / unknown chunk."""
        chunk = self.awaiting.pop(tuple(key), None)
        if chunk is None:
            return None
        q = self.in_flight.get(chunk.req_id)
        assert q and q[0] is chunk, f"slice_done out of order for {key}: in flight {[c.key for c in q or ()]}"
        q.popleft()
        if not q:
            del self.in_flight[chunk.req_id]
        self.stats.done += 1
        return chunk

    def abort_active(self, reason: str) -> list[_Chunk]:
        """Drop the chunks still on stage 0/1 (no decoder left): their requests continue on the normal path from
        the last committed chunk. Chunks awaiting slice_done are older than any active chunk of the same request,
        so what remains in flight stays a contiguous prefix. Returns the dropped chunks."""
        dropped = [c for c in (self.s1, self.s0) if c is not None]
        for chunk in dropped:
            q = self.in_flight[chunk.req_id]
            q.remove(chunk)
            if not q:
                del self.in_flight[chunk.req_id]
            self.pending_aborts.append(chunk.key)
        self.s0 = self.s1 = None
        if dropped:
            self.stats.aborted += len(dropped)
            self.stats.by_reason[reason] = self.stats.by_reason.get(reason, 0) + len(dropped)
        return dropped

    def drop_request(self, req_id: str, reason: str) -> None:
        """Preemption / abort / finish: every in-flight chunk of the request is aborted; blocks are freed by the
        caller (kv_cache_manager.free)."""
        q = self.in_flight.pop(req_id, None)
        for c in q or ():
            self.pending_aborts.append(c.key)
            self.awaiting.pop(c.key, None)
            self.stats.aborted += 1
            self.stats.by_reason[reason] = self.stats.by_reason.get(reason, 0) + 1
        if self.s0 is not None and self.s0.req_id == req_id:
            self.s0 = None
        if self.s1 is not None and self.s1.req_id == req_id:
            self.s1 = None
        self.unseen.discard(req_id)
        self.pending_blocks.pop(req_id, None)

    def forget(self, req_id: str) -> None:
        """The request left the scheduler (finished and freed)."""
        self.drop_request(req_id, "finish")
        self.next_chunk_id.pop(req_id, None)
        self.normal_ok.discard(req_id)

    def take_aborts(self) -> tuple[tuple[str, int], ...]:
        out, self.pending_aborts = tuple(self.pending_aborts), []
        return out


# ---------------------------------------------------------------------------------------------- scheduler glue
def make_planner(vllm_config: Any, chunk_tokens: int) -> SlicePlanner:
    """Stage layer ranges from the model's PP2 partition (``get_pp_indices``, honours VLLM_PP_LAYER_PARTITION)."""
    from vllm.distributed.utils import get_pp_indices

    num_layers = vllm_config.model_config.hf_text_config.num_hidden_layers
    policy = AdaptiveSlicePolicy() if ds41_slices.slice_policy() == "adaptive" else None
    return SlicePlanner(get_pp_indices(num_layers, 0, 2), get_pp_indices(num_layers, 1, 2), chunk_tokens,
                        min_tail=min(MIN_TAIL, chunk_tokens // 2), policy=policy)


def stage_next(sched: Any, planner: SlicePlanner, timestamp: float) -> _Chunk | None:
    """Stage one chunk if stage 0 is free: the next chunk of a running prompt (continuing a staged one first, then
    FCFS), else the first fresh long prompt in the waiting queues. Allocates the chunk's KV blocks without hashing
    them; never preempts (no room -> nothing is staged this step)."""
    from vllm.v1.engine import EngineCoreEventType
    from vllm.v1.request import RequestStatus

    if not planner.can_stage():
        return None
    kv = sched.kv_cache_manager
    running = sorted(sched.running, key=lambda r: not planner.is_staged(r.request_id))   # stable: staged first
    for request in running:
        if not planner.eligible(request) or request.is_finished():
            continue
        if not planner.is_staged(request.request_id) and is_decoding(request):
            continue
        start = planner.staged_upto(request)
        n = planner.chunk_len(request, start)
        if n <= 0:
            continue
        new_blocks = kv.allocate_slots(request, start + n - request.num_computed_tokens, delay_cache_blocks=True)
        if new_blocks is None:
            return None
        unseen = request.request_id in planner.unseen
        return planner.add_chunk(request, start, n, unseen=unseen, new_blocks=None if unseen else new_blocks)

    if len(sched.running) >= sched.max_num_running_reqs:
        return None
    for queue in (sched.waiting, sched.skipped_waiting):
        for request in list(queue):
            if (request.status != RequestStatus.WAITING or request.num_computed_tokens != 0
                    or request.request_id in planner.normal_ok or not planner.wants_stage(request)):
                continue
            computed_blocks, hits = kv.get_computed_blocks(request)
            n = planner.chunk_len(request, hits)
            if n <= 0:
                planner.normal_ok.add(request.request_id)     # the rest fits one normal chunk
                continue
            new_blocks = kv.allocate_slots(
                request, n, num_new_computed_tokens=hits, new_computed_blocks=computed_blocks,
                delay_cache_blocks=True, full_sequence_must_fit=sched.scheduler_reserve_full_isl)
            if new_blocks is None:
                return None
            queue.remove_request(request)
            if request.prefill_stats is not None:
                request.prefill_stats.set(num_prompt_tokens=request.num_prompt_tokens,
                                          num_local_cached_tokens=hits, num_external_cached_tokens=0)
            request.status = RequestStatus.RUNNING
            request.num_computed_tokens = hits
            sched.running.append(request)
            if sched.log_stats:
                request.record_event(EngineCoreEventType.SCHEDULED, timestamp)
            return planner.add_chunk(request, hits, n, unseen=True, new_blocks=None)
    return None
