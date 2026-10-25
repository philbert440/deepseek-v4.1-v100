# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_DS41_SLICE_POLICY=adaptive (hol-design Phase 2): slice sizes from measured worker costs. CPU only.

Policy unit tests, then the real Scheduler + EngineCore against the protocol-checking two-stage fake of
test_ds41_sliced_prefill.py, extended with a simulated clock that matches the merged worker (rivet/hol-p1): stage 1
runs its slices AFTER the step's sampled output and their slice_done rides the next output; the fake reports the
measured costs the way the workers do (stage 0's samples with the chunk handoff, the last stage's with each output).
Run with -s for the table.
"""

from __future__ import annotations

import dataclasses
from collections import deque

import pytest

from tests.v1.core.test_ds41_sliced_prefill import S1, TimedSlicedExecutor, run
from vllm.v1.core.sched.ds41_sliced_prefill import AdaptiveSlicePolicy, SlicePlanner

pytestmark = pytest.mark.cpu_test


# ------------------------------------------------------------------------------------------------ policy
def test_policy_budget_and_carry():
    p = AdaptiveSlicePolicy(keep=1.0, margin_ms=3.0)
    # priors: span 70, stage-0 fwd 32.6 -> budget 34.4 ms -> 2.3 layers at 14.8 ms
    assert p.budget(0) == pytest.approx(70 - 32.6 - 3)
    def chunk_plan(pol, stage=0, lo=0, hi=20):
        a, sizes = lo, []
        while a < hi:
            n, carry = pol.choose(stage, hi - a, a == lo, pol.budget(stage))
            pol.carry[stage] = carry
            sizes.append(n)
            a += n
        return sizes

    sizes = chunk_plan(p)             # x = 2.32 layers: 9 near-equal slices of 2-3, no 1-layer leftover
    assert sum(sizes) == 20 and len(sizes) == round(20 / (34.4 / 14.8)) and set(sizes) <= {2, 3}
    # clamps: at least one layer, never past the stage end
    assert p.choose(0, 20, False, -50.0)[0] == 1
    assert p.choose(0, 2, False, 500.0) == (2, 0.0)
    # keep < 1 stretches the cycle: 70 / 0.75 = 93.3 -> 57.7 ms -> 3.9 layers -> 5 slices of 4
    q = AdaptiveSlicePolicy(keep=0.75, margin_ms=3.0)
    assert chunk_plan(q) == [4] * 5


def test_policy_learns_from_feedback():
    p = AdaptiveSlicePolicy(keep=1.0, margin_ms=0.0, alpha=0.5)
    for _ in range(20):
        p.observe(slice_ms=[(0, 3, 30.0, False), (0, 3, 36.0, True), (1, 19, 57.0, False), (1, 1, 25.0, True)],
                  decode_ms=[(0, 30.0, 0.0, False), (1, 40.0, 80.0, True), (1, 40.0, 500.0, False)])
    assert p.per_layer[0].value == pytest.approx(10.0) and p.head[0].value == pytest.approx(6.0, abs=0.1)
    assert p.per_layer[1].value == pytest.approx(3.0) and p.head[1].value == pytest.approx(22.0, abs=0.1)
    assert p.span.value == pytest.approx(80.0)                  # unclean spans (500 ms) ignored
    assert p.budget(0) == pytest.approx(50.0) and p.budget(1) == pytest.approx(40.0)
    # a first stage-1 slice pays its head before per-layer
    assert p.choose(1, 20, True, p.budget(1))[0] == 6         # (40 - 22) / 3 = 6 layers -> round(20 / 6) = 3 slices


def test_fixed_policy_ignores_feedback_and_keeps_ranges():
    planner = SlicePlanner((0, 20), (20, 40), 256, min_tail=32)
    assert planner.policy is None
    planner.observe([(0, 3, 1.0, False)], [(1, 1.0, 1.0, True)])
    assert planner.fixed_next[0] == {0: 3, 3: 6, 6: 9, 9: 12, 12: 15, 15: 18, 18: 20}


# ------------------------------------------------------------------------------------------------ simulated clock
# Measured (hol-design Appendix A, hol-phase0 arm A): decode verify step stage 0 fwd 32.6 ms, stage 1 fwd 30 + post
# 6.8 ms (span ~70 ms); stage-0 prefill 14.8 ms/layer. Stage-1 split of a chunk is not measured (GPU p50 18.7 ms/layer
# over both slice kinds): two assumptions for layer 20 alone / each of 21-39 under replay.
DEC0, DEC1 = 32.6e-3, 36.8e-3
PRE0, PRE1 = 0.295e-3, 0.228e-3
LAYER0 = 14.8e-3
S1_MODELS = {"s1 A (L20 14.8, 3.2/layer)": (14.8e-3, 60e-3 / 19), "s1 B (L20 30, 10.4/layer)": (30e-3, 198e-3 / 19)}
AL = 69.5 * (DEC0 + DEC1)


class FeedbackTimedExecutor(TimedSlicedExecutor):
    """Two FIFO stages as on the merged worker: per step stage 0 runs decode (+ normal prefill), isends, then its
    slices; stage 1 runs decode + post, the output is ready, THEN its slices (a first slice waits for stage 0's
    handoff); slice_done rides the next output. Cost samples are reported like the workers do."""

    def __init__(self, scale, s1_model, **kw):
        super().__init__(scale, (0.0, 0.0), **kw)
        self.l20, self.pl1 = s1_model
        self.fb: deque = deque()                 # per submitted step: (slice_ms, decode_ms) in FIFO order
        self.s0_buffer: list = []                # stage-0 samples waiting for the next handoff
        self.carry_done: tuple = ()
        self.carry_fb = ((), ())
        self.prev_slices = False
        self.layers_per_slice = {0: [], 1: []}

    def _slice_cost(self, d):
        n = d.layer_end - d.layer_start
        if d.stage == 0:
            return LAYER0 * n
        return (self.l20 if d.layer_start == S1[0] else 0.0) + self.pl1 * (n - (d.layer_start == S1[0]))

    def execute_model(self, so, non_block=False):
        fut = super(TimedSlicedExecutor, self).execute_model(so, non_block)    # FakePPExecutor (no timing)
        if not so.total_num_scheduled_tokens:
            self.end_at[fut.idx] = self.now
            self.fb.append(((), ()))
            return fut
        spec = so.scheduled_spec_decode_tokens
        decode = any(r in spec or (r.startswith("d") and n <= 6) for r, n in so.num_scheduled_tokens.items())
        pre = self.scale * sum(n for r, n in so.num_scheduled_tokens.items() if not r.startswith("d"))
        d0 = [d for d in so.ds41_slices if d.stage == 0]
        d1 = [d for d in so.ds41_slices if d.stage == 1]
        for d in so.ds41_slices:
            self.layers_per_slice[d.stage].append(d.layer_end - d.layer_start)
        t0 = max(self.now, self.free[0])
        d0_end = t0 + (DEC0 if decode else 0.0) + PRE0 * pre
        self.free[0] = d0_end + sum(self._slice_cost(d) for d in d0)
        t1 = max(d0_end, self.free[1])
        d1_end = t1 + (DEC1 if decode else 0.0) + PRE1 * pre
        s1_start = max(d1_end, self.free[0]) if any(d.first for d in d1) else d1_end
        self.free[1] = s1_start + sum(self._slice_cost(d) for d in d1)
        self.end_at[fut.idx] = d1_end                       # sampled output before the stage-1 slices
        # cost feedback as the workers report it
        slice_ms, decode_ms = [], []
        self.s0_buffer += [(0, d.layer_end - d.layer_start, 1e3 * self._slice_cost(d), d.first) for d in d0]
        if decode:
            self.s0_buffer.append(("dec", 1e3 * DEC0))
            clean = not so.ds41_slices and not self.prev_slices
            decode_ms.append((1, 1e3 * DEC1, 1e3 * (d1_end - self.now), clean))
        self.prev_slices = bool(so.ds41_slices)
        if any(d.last for d in d0):                         # the handoff carries stage 0's samples
            decs = [x[1] for x in self.s0_buffer if x[0] == "dec"]
            slice_ms += [x for x in self.s0_buffer if x[0] != "dec"]
            if decs:
                decode_ms.append((0, sum(decs) / len(decs), 0.0, False))
            self.s0_buffer = []
        slice_ms += [(1, d.layer_end - d.layer_start, 1e3 * self._slice_cost(d), d.first) for d in d1]
        self.fb.append((tuple(slice_ms), tuple(decode_ms)))
        return fut

    def _execute(self, so):
        result = super()._execute(so)
        fb = self.fb.popleft()
        if self.pending_output is None:
            # an empty step's output: keep this step's done / samples for the next sampled output
            self.carry_fb = (self.carry_fb[0] + fb[0], self.carry_fb[1] + fb[1])
            return result
        out = self.pending_output
        done_now = tuple(out.ds41_slice_done)
        self.pending_output = dataclasses.replace(
            out, ds41_slice_done=self.carry_done,                       # stage-1 slices ran after this output
            ds41_slice_ms=self.carry_fb[0], ds41_decode_ms=self.carry_fb[1])
        self.carry_done, self.carry_fb = done_now, fb                   # their samples complete later too
        return result


def _sim(monkeypatch, *, policy, s1_model, s0_per=3, keep=None):
    for k in ("VLLM_DS41_SLICE_POLICY", "VLLM_DS41_SLICE_DECODE_KEEP"):
        monkeypatch.delenv(k, raising=False)
    scale = 16
    ex = FeedbackTimedExecutor(scale, s1_model)
    hooks = {}
    if policy == "adaptive":
        hooks["policy"] = AdaptiveSlicePolicy(keep=keep, margin_ms=3.0)
    st = run(monkeypatch, sliced=policy != "lane", lane=policy == "lane", executor=ex, long_prompt=2000, chunk=64,
             min_tail=8, dec_max_tokens=1500, max_model_len=2048, inject_after=4, s0_per=s0_per, **hooks)
    info = st["info"]
    t0, t1 = info["t_inject"], info["t_first"]
    dec = [t for t, rid in info["arrivals"] if rid == "d0" and t0 < t <= t1]
    gaps = [b - a for a, b in zip([t0] + dec, dec + [t1])]
    lps = ex.layers_per_slice
    return {"dec_tps": AL * len(dec) / (t1 - t0), "prefill_tps": scale * 2000 / (t1 - t0), "max_gap": max(gaps),
            "s0_layers": sum(lps[0]) / max(len(lps[0]), 1), "s1_slices": len(lps[1])}


def test_timeline_adaptive_vs_fixed(monkeypatch):
    rows = {}
    for mname, model in S1_MODELS.items():
        rows[(mname, "fixed S0=3")] = _sim(monkeypatch, policy="fixed", s1_model=model, s0_per=3)
        rows[(mname, "fixed S0=4")] = _sim(monkeypatch, policy="fixed", s1_model=model, s0_per=4)
        for keep in (1.0, 0.75, 0.7):
            rows[(mname, f"adaptive keep={keep}")] = _sim(monkeypatch, policy="adaptive", s1_model=model, keep=keep)
    rows[("-", "phase-0 lane, no slicing")] = _sim(monkeypatch, policy="lane", s1_model=next(iter(S1_MODELS.values())))
    print(f"\n32K prefill + one B=1 decoder (idle decode 69.5 tok/s, AL {AL:.2f}); stage-1 slices after the output")
    for (m, k), r in rows.items():
        print(f"  {m:27s} {k:26s} decoder {r['dec_tps']:5.1f} tok/s  prefill {r['prefill_tps']:5.0f} tok/s  "
              f"max gap {1e3 * r['max_gap']:4.0f} ms  s0 layers/slice {r['s0_layers']:.2f}")
    a = rows[(next(iter(S1_MODELS)), "adaptive keep=0.75")]
    assert a["dec_tps"] >= 40 and a["prefill_tps"] >= 2000, a
    gap_fit = rows[(next(iter(S1_MODELS)), "adaptive keep=1.0")]
    assert gap_fit["dec_tps"] > rows[(next(iter(S1_MODELS)), "fixed S0=3")]["dec_tps"]
