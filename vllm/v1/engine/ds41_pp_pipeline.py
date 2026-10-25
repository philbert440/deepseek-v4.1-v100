# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pipelined chunked prefill under PP + synchronous scheduling (``VLLM_DS41_PP_PREFILL_PIPELINE``; default 0 = off).

Why one PP stage idles during a long prefill today: with PP the engine runs ``step_with_batch_queue`` with a queue of
``pp_size`` batches, and the scheduler already advances ``num_computed_tokens`` at schedule time, so chunk k+1 of a
request is schedulable while chunk k is in flight (scheduler.py ``_update_after_schedule`` / the ``num_new_tokens == 0``
"PP>1" note). But with speculative decoding (DSpark) and synchronous scheduling, ``EngineCore.post_step`` calls
``take_draft_token_ids`` after EVERY step. That RPC is blocking and its future drains every earlier future of the
output rank (``FutureWrapper.result``), so the engine thread waits until the step it just submitted has finished on the
last stage before it schedules the next one: exactly one batch is ever in flight.

With the knob on, ``post_step`` skips that blocking call after a step in which every scheduled request is a partial
prefill chunk (no sampled token is kept, no drafts are proposed for it: ``update_draft_token_ids`` clears the drafts of
prefill chunks anyway), so the next chunk goes to stage 0 while stage 1 still works on the previous one. Steps with a
decode / verify row, a final prefill chunk, structured-output deferral or DDTree payloads keep today's path. The order of
scheduler calls (``schedule`` / ``update_from_output`` / ``update_draft_token_ids``) is the same as with the knob off --
only the partial steps' no-op draft updates are dropped -- so step compositions (and with them the fixed-M GEMM buckets)
do not change. At most ``pp_size`` batches are in flight (the batch queue's size).

Worker side (``gpu_worker.py``): a non-last PP rank may now start preparing step k+1 while its own GPU still runs step k.
With the knob on, it waits (CPU) for an event recorded after step k's forward before it prepares step k+1, so the
runner's reused pinned buffers, the DS41 metadata / demand-KV / replay host state never race the GPU; the cross-stage
overlap is untouched (stage 0 can run k+1 while stage 1 runs k).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vllm.models.deepseek_v41 import knobs

PP_PREFILL_PIPELINE_KNOB = "VLLM_DS41_PP_PREFILL_PIPELINE"


def pp_prefill_pipeline_enabled() -> bool:
    """The knob (strict boolean parse; unset = off)."""
    return knobs.env_bool(PP_PREFILL_PIPELINE_KNOB, False)


def is_partial_prefill_only(scheduler_output: Any, requests: Mapping[str, Any]) -> bool:
    """True when every request of the (just scheduled) step is a partial prefill chunk.

    Must be evaluated right after ``Scheduler.schedule()`` returned this output: ``Request.is_prefill_chunk`` is set
    by ``_update_after_schedule`` and is overwritten when the request is scheduled again.
    """
    num_scheduled = scheduler_output.num_scheduled_tokens
    if not num_scheduled or scheduler_output.total_num_scheduled_tokens <= 0:
        return False
    if scheduler_output.scheduled_spec_decode_tokens:
        return False
    if getattr(scheduler_output, "pending_structured_output_tokens", False):
        return False
    if getattr(scheduler_output, "scheduled_ddtree_payloads", None):
        return False
    for req_id in num_scheduled:
        request = requests.get(req_id)
        if request is None or not request.is_prefill_chunk:
            return False
    return True
