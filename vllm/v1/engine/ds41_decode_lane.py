# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decode lane for the PP batch queue (``VLLM_DS41_DECODE_LANE``; default 0 = off). hol-design.md §2(b) / §3 Phase 0.

Today (PP2, sync scheduling, batch queue of 2): while a decoder's step is in flight the scheduler cannot schedule the
decoder again (its next tokens depend on that step), so the second batch-queue slot gets a prefill-only chunk. Workers
run RPCs FIFO, so the decoder's next step waits behind that chunk on stage 0: the decoder rides every OTHER step and
its token period is ~2 x max(stage-0, stage-1) step cost.

With the lane on, ``EngineCore.step_with_batch_queue`` does not schedule into the second slot while every decoding
request already has a batch in flight; it blocks on the oldest batch instead. Every submitted step then carries the
decode/verify rows (plus whatever prefill chunk the scheduler adds), and the decoder's token period becomes
stage-0 + stage-1 of one step. The decoder speedup is therefore 2*max(s0, s1) / (s0 + s1) -- ~2x when one stage
dominates (cold Engram I/O on stage 0), ~1x when the stages are balanced -- and while a decoder exists a long prefill
loses its stage overlap (T per s0+s1 instead of T per max(s0, s1)). With no decoder the lane never holds, so prefill
pipelining (``VLLM_DS41_PP_PREFILL_PIPELINE``) is untouched; with B>=2 decoders a decoder that is not in flight is
still scheduled into the second slot (that batch carries its decode rows), so today's B=2 overlap is kept.

The lane requires ``VLLM_DS41_FOLD_DRAFTS=1``: without it ``post_step`` issues the blocking ``take_draft_token_ids``
RPC after every decode step, which drains the queue anyway and the lane would only remove the prefill-only steps.
The scheduler is never asked to undo a step: the hold is decided before ``schedule()`` runs, so lane off is
byte-for-byte today's engine loop.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from vllm.models.deepseek_v41 import knobs

DECODE_LANE_KNOB = "VLLM_DS41_DECODE_LANE"
FOLD_DRAFTS_KNOB = "VLLM_DS41_FOLD_DRAFTS"


def decode_lane_enabled() -> bool:
    """The knob (strict boolean parse; unset = off)."""
    return knobs.env_bool(DECODE_LANE_KNOB, False)


def check_config(*, fold_drafts: bool, async_scheduling: bool, knob: str = DECODE_LANE_KNOB) -> None:
    """Refuse a lane configuration that cannot work (called once at engine start when the knob is on; the
    layer-sliced prefill knob reuses the lane and the same checks)."""
    if not fold_drafts:
        raise ValueError(
            f"{knob}=1 requires {FOLD_DRAFTS_KNOB}=1: without it post_step's blocking "
            "take_draft_token_ids RPC drains the batch queue after every decode step")
    if async_scheduling:
        raise ValueError(f"{knob}=1 supports synchronous scheduling only (--no-async-scheduling)")


def is_decoding(request: Any) -> bool:
    """Past its prompt at schedule-time accounting: decoding, or its final prefill chunk is in flight (that step
    samples and proposes drafts, so it is a decode step for the lane)."""
    return request.num_computed_tokens >= request.num_prompt_tokens


def should_hold(batch_queue: Iterable[tuple], running: Iterable[Any]) -> bool:
    """True when the next batch could only be prefill: at least one request is decoding and every decoding request
    has a batch in flight. ``batch_queue`` entries are ``(future, scheduler_output, exec_future)``."""
    in_flight: set[str] | None = None
    has_decoder = False
    for request in running:
        if not is_decoding(request):
            continue
        has_decoder = True
        if in_flight is None:
            in_flight = set()
            for entry in batch_queue:
                in_flight.update(entry[1].num_scheduled_tokens)
        if request.request_id not in in_flight:
            return False   # this decoder can be scheduled now: the batch will carry its decode rows
    return has_decoder
