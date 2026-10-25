# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark per-step trace for the DeepSeek-V4.1 SM70 port (``VLLM_DS41_DSPARK_TRACE``; default unset = off).

Debug instrumentation for the "pos0 acceptance capped at ~0.49" investigation.
With the knob set to a file path, the LAST pipeline stage's TP rank 0 appends one JSON line per request per
``sample_tokens`` step that ran the drafter:

  step        trace counter on this worker (one per sample_tokens call that reached the trace)
  req         request id
  computed    num_computed_tokens at the start of the step = position of the step's row 0
  sched       scheduled tokens of the request this step (1 + drafts for decode, the chunk for prefill)
  spec_in     the draft tokens this step VERIFIED (scheduler_output.scheduled_spec_decode_tokens), [] if none
  out         the valid sampled tokens of the step (accepted drafts + bonus; [] for a partial prefill chunk)
  ctx         [first, last] context position the drafter absorbed for this request (target positions of its rows)
  anchor_pos  the drafter's query-row-0 position (must be computed + len(out) for a decode step)
  anchor_tok  the drafter's query-row-0 token (must be out[-1])
  drafts      the drafts the drafter proposed this step (verified by the request's NEXT step)
  top5        drafter pos0 top-5 [token, logprob] (before the Markov bias of later positions), or null
  nonfinite   positions whose tap row was non-finite this step (VLLM_DS41_DSPARK_TAP_SHIFT>0 zeroes them), or null

``scripts/dspark_trace_analyze.py`` checks the invariants (anchor = last accepted + 1, the next step verifies exactly
these drafts, positions advance by len(out)) and reports pos0 acceptance conditioned on the previous step's accepted
count, the alternation statistic and an off-by-one (shift) test.

Cost when on: a few small device->host copies and one host sync per step (debug only). Off: one getattr per step.
"""

from __future__ import annotations

import json
import os
import threading
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)

TRACE_KNOB = "VLLM_DS41_DSPARK_TRACE"
TRACE_MAX_KNOB = "VLLM_DS41_DSPARK_TRACE_MAX"      # lines cap (default 200000)
# hand-off from the drafter model (sm70/dspark_taps.note_nonfinite) to the runner-side trace of the same step:
# {"nonfinite_pos": [positions whose main_x row was non-finite and zeroed]}
LAST_CONTEXT: dict = {}


def trace_path() -> str | None:
    value = os.environ.get(TRACE_KNOB, "").strip()
    return value or None


def _trace_max() -> int:
    raw = os.environ.get(TRACE_MAX_KNOB, "").strip()
    if not raw:
        return 200000
    value = int(raw)
    if value < 0:
        raise ValueError(f"{TRACE_MAX_KNOB}={raw}: must be >= 0")
    return value


class DSparkTrace:
    """Owned by the model runner on the last PP stage, TP rank 0 (see ``for_runner``)."""

    def __init__(self, path: str, max_lines: int | None = None) -> None:
        self.path = path
        self.max_lines = _trace_max() if max_lines is None else max_lines
        self.lines = 0
        self.step = 0
        self._lock = threading.Lock()
        self._failed = False
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        logger.info("DSpark trace on: %s (max %d lines)", path, self.max_lines)

    @staticmethod
    def for_runner(runner: Any) -> "DSparkTrace | None":
        path = trace_path()
        if path is None or getattr(runner, "speculative_config", None) is None:
            return None
        try:
            from vllm.distributed.parallel_state import get_pp_group, get_tp_group

            if not get_pp_group().is_last_rank or get_tp_group().rank_in_group != 0:
                return None
            pp_rank = get_pp_group().rank_in_group
        except (AssertionError, AttributeError, RuntimeError):     # tests / single process
            pp_rank = 0
        root, ext = os.path.splitext(path)
        return DSparkTrace(f"{root}.pp{pp_rank}{ext or '.jsonl'}")

    def record(self, records: list[dict]) -> None:
        if not records or self.lines >= self.max_lines:
            return
        with self._lock, open(self.path, "a") as f:
            for rec in records:
                if self.lines >= self.max_lines:
                    break
                f.write(json.dumps(rec, separators=(",", ":")) + "\n")
                self.lines += 1

    def collect(self, *, req_ids: list[str], computed: list[int], sched: dict[str, int],
                spec_in: dict[str, list[int]], out: list[list[int]], ctx_qsl: list[int] | None,
                ctx_positions: list[int] | None, anchor_pos: list[int] | None, anchor_tok: list[int] | None,
                drafts: list[list[int]] | None, top5: list[list[list[float]]] | None,
                nonfinite: list[int] | None = None) -> list[dict]:
        """Pure host-side assembly (unit-tested): one record per request of this step."""
        step = self.step
        self.step += 1
        recs = []
        for i, rid in enumerate(req_ids):
            ctx = None
            if ctx_qsl is not None and ctx_positions is not None and i + 1 < len(ctx_qsl):
                a, b = int(ctx_qsl[i]), int(ctx_qsl[i + 1])
                if b > a and b <= len(ctx_positions):
                    ctx = [int(ctx_positions[a]), int(ctx_positions[b - 1])]
            recs.append({
                "step": step,
                "req": rid,
                "computed": int(computed[i]),
                "sched": int(sched.get(rid, 0)),
                "spec_in": [int(t) for t in spec_in.get(rid, ()) or ()],
                "out": [int(t) for t in out[i]] if i < len(out) else [],
                "ctx": ctx,
                "anchor_pos": None if anchor_pos is None or i >= len(anchor_pos) else int(anchor_pos[i]),
                "anchor_tok": None if anchor_tok is None or i >= len(anchor_tok) else int(anchor_tok[i]),
                "drafts": None if drafts is None or i >= len(drafts) else [int(t) for t in drafts[i]],
                "top5": None if top5 is None or i >= len(top5) else top5[i],
                "nonfinite": nonfinite,
            })
        return recs

    def on_step(self, runner: Any, scheduler_output: Any, valid_sampled_token_ids: list[list[int]],
                spec_decode_common_attn_metadata: Any) -> None:
        """Called from GPUModelRunner.sample_tokens after the drafter and the bookkeeping ran. Never raises."""
        if self._failed or self.lines >= self.max_lines:
            return
        try:
            self._on_step(runner, scheduler_output, valid_sampled_token_ids, spec_decode_common_attn_metadata)
        except Exception:                      # debug instrumentation must not take the server down
            self._failed = True
            logger.exception("DSpark trace failed; trace disabled for the rest of the run")

    def _on_step(self, runner, scheduler_output, valid_sampled_token_ids, cad) -> None:
        drafter = getattr(runner, "drafter", None)
        if drafter is None or cad is None:
            return
        ib = runner.input_batch
        n = int(ib.num_reqs)
        req_ids = list(ib.req_ids[:n])
        q = int(getattr(drafter, "num_query_per_req", 0) or 0)
        anchor_pos = anchor_tok = ctx_positions = None
        if q > 0:
            idx = [i * q for i in range(n)]
            anchor_pos = drafter.positions[idx].tolist() if drafter.positions.dim() == 1 else None
            anchor_tok = drafter.input_ids[idx].tolist()
        num_ctx = int(getattr(drafter, "_dflash_num_context", 0) or 0)
        buf = getattr(drafter, "_context_positions_buffer", None)
        if buf is not None and num_ctx > 0:
            ctx_positions = buf[:num_ctx].tolist()
        qsl = cad.query_start_loc_cpu[: n + 1].tolist()
        drafts = None
        if getattr(runner, "_draft_token_req_ids", None):
            d, _ = runner._get_draft_token_ids_cpu()
            drafts = d
        top5 = getattr(drafter, "_ds41_trace_top5", None)
        if top5 is not None:
            ids, lps = top5
            top5 = [[[int(t), round(float(v), 3)] for t, v in zip(r_ids, r_lps)]
                    for r_ids, r_lps in zip(ids.tolist(), lps.tolist())]
            drafter._ds41_trace_top5 = None
        recs = self.collect(
            req_ids=req_ids,
            computed=[int(x) for x in ib.num_computed_tokens_cpu[:n]],
            sched=dict(scheduler_output.num_scheduled_tokens),
            spec_in={k: list(v) for k, v in scheduler_output.scheduled_spec_decode_tokens.items()},
            out=[list(x) for x in valid_sampled_token_ids],
            ctx_qsl=qsl,
            ctx_positions=ctx_positions,
            anchor_pos=anchor_pos,
            anchor_tok=anchor_tok,
            drafts=drafts,
            top5=top5,
            nonfinite=LAST_CONTEXT.pop("nonfinite_pos", None),
        )
        self.record(recs)


def stash_pos0_top5(proposer: Any, step_logits: Any) -> None:
    """DSparkProposer._sample_draft_tokens, position 0, trace on: keep the drafter's pos0 top-5 (GPU tensors; the
    runner-side trace copies them to the host after the step). Skipped under CUDA-graph capture."""
    import torch

    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        return
    lp = torch.log_softmax(step_logits.float(), dim=-1)
    vals, ids = torch.topk(lp, k=min(5, lp.shape[-1]), dim=-1)
    m = getattr(getattr(proposer, "model", None), "map_draft_to_target", None)
    if m is not None:
        ids = m(ids)
    proposer._ds41_trace_top5 = (ids, vals)
