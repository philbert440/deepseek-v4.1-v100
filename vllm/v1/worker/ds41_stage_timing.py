# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-PP-stage step timing in the serve log (``VLLM_DS41_STAGE_TIMING_EVERY=N``; default 0 = off).

``MixedPrefillTimer`` (mixed_prefill.py) runs on the output rank only, so the cost of a step on the earlier stages is
invisible (hol-design.md §3 Phase 0). This timer runs on TP rank 0 of EVERY PP stage and logs, every N timed steps,
one line per stage with the step cost split by step kind (decode-only / mixed / prefill-only):

  fwd   GPU time of this stage's forward, from the moment the previous stage's activations arrived (the PP irecv
        completed on the compute stream; stage 0: the step's first queued GPU work) to the end of the forward
  post  last stage only: sampling + drafter + draft take after the forward (sample_tokens)
  span  first queued GPU work of the step -> end, i.e. fwd + post + the wait for the previous stage
  gap   host time between consecutive execute_model calls on this rank (the stage's step cadence)

CUDA events only, polled without synchronizing (a step is logged once its events have completed), so the knob does not
change scheduling or numerics. Off by default; the worker never constructs the timer when N=0.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from vllm.models.deepseek_v41 import knobs

STAGE_TIMING_KNOB = "VLLM_DS41_STAGE_TIMING_EVERY"
_MAX_PENDING = 64
KINDS = ("decode", "mixed", "prefill")


def stage_timing_every() -> int:
    return knobs.env_int(STAGE_TIMING_KNOB, 0, minimum=0)


def classify(scheduler_output: Any) -> tuple[str, int, int, int]:
    """(kind, decode_rows, decode_tokens, prefill_tokens) of a scheduled step. A row is a decode row when it carries
    drafts, or schedules a single token of a request the worker already holds."""
    spec = scheduler_output.scheduled_spec_decode_tokens or {}
    new_ids = {r.req_id for r in scheduler_output.scheduled_new_reqs}
    rows = dec_tok = 0
    for req_id, n in scheduler_output.num_scheduled_tokens.items():
        if req_id in spec or (n == 1 and req_id not in new_ids):
            rows += 1
            dec_tok += n
    pre_tok = scheduler_output.total_num_scheduled_tokens - dec_tok
    kind = "mixed" if rows and pre_tok else ("decode" if rows else "prefill")
    return kind, rows, dec_tok, pre_tok


@dataclass
class _Step:
    kind: str
    rows: int
    dec_tok: int
    pre_tok: int
    gap_ms: float
    ev_begin: Any
    ev_recv: Any = None
    ev_fwd: Any = None
    ev_end: Any = None
    had_slices: bool = False


@dataclass
class _Acc:
    n: int = 0
    rows: int = 0
    dec_tok: int = 0
    pre_tok: int = 0
    fwd: float = 0.0
    fwd_max: float = 0.0
    post: float = 0.0
    span: float = 0.0
    gap: float = 0.0


@dataclass
class StageStepTimer:
    every: int
    pp_rank: int
    is_last: bool
    log: Callable[..., None]
    event_factory: Callable[[], Any] | None = None
    # (kind, had_slices, fwd_ms, post_ms, span_ms) per completed step (adaptive slice sizing feedback); every=0 with
    # a callback = measure without logging
    on_step: Callable[..., None] | None = None
    clock: Callable[[], float] = time.perf_counter
    _cur: _Step | None = None
    _pending: deque = field(default_factory=deque)
    _acc: dict = field(default_factory=dict)
    _timed: int = 0
    _last_begin: float | None = None
    _last_done: _Step | None = None
    _last_ms: tuple = (0.0, 0.0, 0.0)

    def _event(self):
        if self.event_factory is not None:
            return self.event_factory()
        import torch

        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        return ev

    # ---- hooks, in call order within one step ----
    def begin(self, scheduler_output: Any) -> None:
        self._poll()
        now = self.clock()
        gap = 0.0 if self._last_begin is None else (now - self._last_begin) * 1000.0
        self._last_begin = now
        self._cur = None
        if scheduler_output.total_num_scheduled_tokens <= 0 or len(self._pending) >= _MAX_PENDING:
            return
        kind, rows, dec_tok, pre_tok = classify(scheduler_output)
        self._cur = _Step(kind, rows, dec_tok, pre_tok, gap, self._event(),
                          had_slices=bool(getattr(scheduler_output, "ds41_slices", ())))

    def recv_done(self) -> None:
        """Called from the PP irecv postprocess (after the handles were waited on the compute stream)."""
        if self._cur is not None and self._cur.ev_recv is None:
            self._cur.ev_recv = self._event()

    def fwd_done(self) -> None:
        if self._cur is not None and self._cur.ev_fwd is None:
            self._cur.ev_fwd = self._event()

    def finish(self) -> None:
        cur, self._cur = self._cur, None
        if cur is None:
            return
        if cur.ev_fwd is None:
            cur.ev_fwd = self._event()
        cur.ev_end = self._event() if self.is_last else cur.ev_fwd
        self._pending.append(cur)
        self._poll()

    # ---- accounting ----
    def _poll(self) -> None:
        while self._pending and self._pending[0].ev_end.query():
            st = self._pending.popleft()
            start = st.ev_recv if st.ev_recv is not None else st.ev_begin
            fwd = start.elapsed_time(st.ev_fwd)
            post = st.ev_fwd.elapsed_time(st.ev_end) if st.ev_end is not st.ev_fwd else 0.0
            span = st.ev_begin.elapsed_time(st.ev_end)
            acc = self._acc.setdefault(st.kind, _Acc())
            acc.n += 1
            acc.rows += st.rows
            acc.dec_tok += st.dec_tok
            acc.pre_tok += st.pre_tok
            acc.fwd += fwd
            acc.fwd_max = max(acc.fwd_max, fwd)
            acc.post += post
            acc.span += span
            acc.gap += st.gap_ms
            self._last_done, self._last_ms = st, (fwd, post, span)
            self._timed += 1
            if self.on_step is not None:
                self.on_step(st.kind, st.had_slices, fwd, post, span)
            if self.every and self._timed % self.every == 0:
                self._emit()

    def _emit(self) -> None:
        parts = []
        for kind in KINDS:
            a = self._acc.get(kind)
            if not a or not a.n:
                continue
            s = (f"{kind} n={a.n} rows={a.rows / a.n:.1f} dec_tok={a.dec_tok / a.n:.1f} "
                 f"pre_tok={a.pre_tok / a.n:.0f} fwd={a.fwd / a.n:.1f}ms(max {a.fwd_max:.1f})")
            if a.pre_tok:
                s += f" fwd/1k_pre={1000.0 * a.fwd / a.pre_tok:.1f}ms"
            if self.is_last:
                s += f" post={a.post / a.n:.1f}ms"
            s += f" span={a.span / a.n:.1f}ms gap={a.gap / a.n:.1f}ms"
            parts.append(s)
        st = self._last_done
        fwd, post, span = self._last_ms
        last = (f"last={st.kind}(rows={st.rows} dec_tok={st.dec_tok} pre_tok={st.pre_tok} fwd={fwd:.1f} "
                f"post={post:.1f} span={span:.1f} gap={st.gap_ms:.1f})") if st is not None else ""
        self.log("DS41 stage timing pp_rank=%d%s steps=%d | %s | %s", self.pp_rank,
                 " (last stage)" if self.is_last else "", self._timed, " | ".join(parts), last)
        self._acc = {}
