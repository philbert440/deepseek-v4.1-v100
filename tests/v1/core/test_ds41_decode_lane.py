# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_DS41_DECODE_LANE (hol-design.md Phase 0): no prefill-only batch in the PP batch queue's second slot while a
decoder is in flight; needs VLLM_DS41_FOLD_DRAFTS=1.

CPU only. Exactness and step composition run the real Scheduler through the real ``EngineCore.step_with_batch_queue``
/ ``post_step`` against the FIFO two-stage fake of test_ds41_pp_variable_drafts.py (exact greedy target, random drafts).
The timeline tests add a simulated clock to that fake: stage 0 and stage 1 each run their RPCs FIFO with a cost model
(intercept + per-prefill-token), a step's output is ready when stage 1 finishes it, and the engine thread's clock
advances only when it blocks on a result. Run with -s to print the per-token timelines.
"""

from __future__ import annotations

import random
from types import SimpleNamespace

import pytest

from tests.v1.core.test_ds41_pp_variable_drafts import (
    FIXED,
    RANDOM,
    FakePPExecutor,
    _core,
    _Fut,
    greedy_reference,
)
from tests.v1.core.utils import create_requests, create_scheduler
from vllm.v1.engine import ds41_decode_lane
from vllm.v1.engine.core import EngineCore

pytestmark = pytest.mark.cpu_test

KNOBS = ("VLLM_DS41_DECODE_LANE", "VLLM_DS41_FOLD_DRAFTS", "VLLM_DS41_PP_NO_EMPTY_STEP")


# ----------------------------------------------------------------------------------------- knob, config, predicate
def test_knob_default_off_and_strict(monkeypatch):
    monkeypatch.delenv("VLLM_DS41_DECODE_LANE", raising=False)
    assert ds41_decode_lane.decode_lane_enabled() is False
    monkeypatch.setenv("VLLM_DS41_DECODE_LANE", "1")
    assert ds41_decode_lane.decode_lane_enabled() is True
    monkeypatch.setenv("VLLM_DS41_DECODE_LANE", "0")
    assert ds41_decode_lane.decode_lane_enabled() is False
    monkeypatch.setenv("VLLM_DS41_DECODE_LANE", "maybe")
    with pytest.raises(ValueError):
        ds41_decode_lane.decode_lane_enabled()


def test_lane_requires_fold_drafts():
    with pytest.raises(ValueError, match="VLLM_DS41_FOLD_DRAFTS=1"):
        ds41_decode_lane.check_config(fold_drafts=False, async_scheduling=False)
    with pytest.raises(ValueError, match="synchronous"):
        ds41_decode_lane.check_config(fold_drafts=True, async_scheduling=True)
    ds41_decode_lane.check_config(fold_drafts=True, async_scheduling=False)


def test_engine_init_refuses_lane_without_fold(monkeypatch):
    """The refusal sits in EngineCore.__init__ (after the batch queue is built), before any request runs."""
    import inspect

    src = inspect.getsource(EngineCore.__init__)
    assert "ds41_decode_lane.check_config(" in src
    assert src.index("self.batch_queue = deque(") < src.index("ds41_decode_lane.check_config(")


def _req(rid, computed, prompt):
    return SimpleNamespace(request_id=rid, num_computed_tokens=computed, num_prompt_tokens=prompt)


def _entry(*rids):
    return (None, SimpleNamespace(num_scheduled_tokens={r: 1 for r in rids}), None)


def test_should_hold_cases():
    hold = ds41_decode_lane.should_hold
    dec, pre = _req("d", 120, 100), _req("p", 2048, 16384)
    assert not hold([_entry("p")], [pre])                    # no decoder: prefill pipelining untouched
    assert hold([_entry("d", "p")], [dec, pre])              # decoder in flight: hold the second slot
    assert not hold([_entry("p")], [dec, pre])               # decoder schedulable now: its batch carries it
    dec2 = _req("e", 300, 50)
    assert not hold([_entry("d", "p")], [dec, dec2, pre])    # B=2: the other decoder may take the second slot
    assert hold([_entry("d"), _entry("e", "p")], [dec, dec2, pre])
    final = _req("f", 16384, 16384)                          # final prefill chunk in flight counts as decoding
    assert hold([_entry("f")], [final])
    assert not hold([], [])


# ------------------------------------------------------------------------------------- engine loop, mixed arrivals
def _lane_core(scheduler, executor, lane: bool) -> EngineCore:
    core = _core(scheduler, executor)
    core._ds41_decode_lane = lane
    core._ds41_lane_holds = 0
    return core


def _mixed_run(monkeypatch, *, lane: bool, fold: bool = True, executor=None, decoders=1, long_prompt=1500,
               chunk=256, inject_after=8, max_tokens=None, accuracy=0.8, seed=0, policy=FIXED, max_model_len=4096, bare=False):
    """`decoders` short requests start decoding; once the first has `inject_after` tokens a long prompt arrives.
    Returns (executor, outputs, per-token arrival log, injection info)."""
    for k in KNOBS:
        monkeypatch.delenv(k, raising=False)
    if fold:
        monkeypatch.setenv("VLLM_DS41_FOLD_DRAFTS", "1")
    if lane:
        monkeypatch.setenv("VLLM_DS41_DECODE_LANE", "1")
    sched = create_scheduler(max_num_seqs=decoders + 1, max_num_batched_tokens=chunk, max_model_len=max_model_len,
                             num_speculative_tokens=5, async_scheduling=False)
    object.__setattr__(sched.parallel_config, "pipeline_parallel_size", 2)
    sched.use_pp = True
    ex = executor if executor is not None else FakePPExecutor(policy, accuracy, seed)
    core = _core(sched, ex) if bare else _lane_core(sched, ex, lane)
    rng = random.Random(seed + 1)
    if max_tokens is None:
        max_tokens = 200
    short = create_requests(decoders, num_tokens=37, max_tokens=max_tokens, ignore_eos=True,
                            req_ids=[f"d{i}" for i in range(decoders)])
    (big,) = create_requests(1, num_tokens=long_prompt, max_tokens=4, ignore_eos=True, req_ids=["p"])
    for r in (*short, big):
        r.prompt_token_ids[:] = [rng.randrange(1000) for _ in r.prompt_token_ids]
        r._all_token_ids[:] = list(r.prompt_token_ids)
    for r in short:
        sched.add_request(r)
    outputs = {r.request_id: [] for r in (*short, big)}
    arrivals = []          # (clock, req_id, n_tokens)
    info = {}
    for _ in range(20000):
        if not (sched.has_requests() or core.batch_queue):
            break
        if "t_inject" not in info and len(outputs["d0"]) >= inject_after:
            sched.add_request(big)
            info["t_inject"] = getattr(ex, "now", 0.0)
            info["step_inject"] = len(ex.steps)
        eco_map, executed = core.step_with_batch_queue()
        core.post_step(executed)
        for eco in (eco_map or {}).values():
            for o in eco.outputs:
                outputs[o.request_id].extend(o.new_token_ids)
                if o.new_token_ids:
                    arrivals.append((getattr(ex, "now", 0.0), o.request_id, len(o.new_token_ids)))
                    if o.request_id == "p" and "t_first" not in info:
                        info["t_first"] = getattr(ex, "now", 0.0)
                        info["step_first"] = len(ex.steps)
    else:
        raise AssertionError("engine loop did not finish")
    for r in (*short, big):
        assert outputs[r.request_id] == greedy_reference(r.prompt_token_ids, r.max_tokens), r.request_id
    info["holds"] = getattr(core, "_ds41_lane_holds", 0)
    return ex, outputs, arrivals, info


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("decoders", [1, 2])
def test_lane_exact_and_composition(monkeypatch, seed, decoders):
    """Lane on or off: every request is the exact greedy continuation. Lane off: while a decoder exists the queue
    carries prefill-only steps. Lane on: every step submitted while a decoder exists carries a decode row."""
    kw = dict(decoders=decoders, seed=seed, policy=RANDOM)
    ex_off, out_off, _, info_off = _mixed_run(monkeypatch, lane=False, **kw)
    ex_on, out_on, _, info_on = _mixed_run(monkeypatch, lane=True, **kw)
    assert out_on == out_off
    assert info_off["holds"] == 0 and info_on["holds"] > 0

    def prefill_only_while_decoding(ex, info):
        n = 0
        for step in ex.steps[info["step_inject"]:info["step_first"]]:
            if set(step) == {"p"}:
                n += 1
        return n

    assert prefill_only_while_decoding(ex_off, info_off) > 0
    assert prefill_only_while_decoding(ex_on, info_on) == 0
    assert ex_on.rpcs.count("take") == 0


def test_lane_off_is_todays_loop(monkeypatch):
    """Knob off: same RPC sequence and step compositions as an engine without the lane attribute at all."""
    for fold in (False, True):
        ex_a, out_a, _, _ = _mixed_run(monkeypatch, lane=False, fold=fold, seed=4, policy=RANDOM)
        ex_b, out_b, _, _ = _mixed_run(monkeypatch, lane=False, fold=fold, seed=4, policy=RANDOM, bare=True)
        assert (ex_a.rpcs, ex_a.steps, out_a) == (ex_b.rpcs, ex_b.steps, out_b)


def test_no_decoder_keeps_prefill_pipelining(monkeypatch):
    """A lone long prompt (no decoder yet): the lane never holds, two chunks stay in flight as with the knob off."""
    for lane in (False, True):
        for k in KNOBS:
            monkeypatch.delenv(k, raising=False)
        monkeypatch.setenv("VLLM_DS41_FOLD_DRAFTS", "1")
        sched = create_scheduler(max_num_seqs=1, max_num_batched_tokens=128, max_model_len=4096,
                                 num_speculative_tokens=5, async_scheduling=False)
        object.__setattr__(sched.parallel_config, "pipeline_parallel_size", 2)
        sched.use_pp = True
        ex = FakePPExecutor(FIXED, 0.8, 0)
        core = _lane_core(sched, ex, lane)
        (r,) = create_requests(1, num_tokens=1000, max_tokens=2, ignore_eos=True)
        sched.add_request(r)
        queued_at_submit = []
        orig = ex.execute_model

        def execute_model(so, non_block=False, _orig=orig, _core=core, _log=queued_at_submit):
            _log.append((len(_core.batch_queue), so.num_scheduled_tokens.get(r.request_id, 0)))
            return _orig(so, non_block)

        ex.execute_model = execute_model
        for _ in range(500):
            if not (sched.has_requests() or core.batch_queue):
                break
            core.step_with_batch_queue()
            core.post_step(True)
        # every prefill chunk after the first entered while the previous chunk was still in flight
        chunks = [q for q, n in queued_at_submit if n == 128]
        assert len(chunks) == 7 and chunks[1:] == [1] * 6
        if not lane:
            assert core._ds41_lane_holds == 0


# ------------------------------------------------------------------------------------------- simulated timeline
class _TimedFut(_Fut):
    def done(self) -> bool:
        return self.ex.end_at[self.idx] <= self.ex.now

    def result(self):
        self.ex.now = max(self.ex.now, self.ex.end_at[self.idx])
        return super().result()


class TimedPPExecutor(FakePPExecutor):
    """FakePPExecutor + a clock: two FIFO stages, cost(stage) = base + per_tok * prefill_tokens + per_row * rows."""

    def __init__(self, cost0, cost1, tok_scale=1, **kw):
        super().__init__(FIXED, kw.pop("accuracy", 0.8), kw.pop("seed", 0))
        self.tok_scale = tok_scale     # opt-125m caps max_model_len at 2048: a 128-token chunk stands for 1024
        self.cost = (cost0, cost1)
        self.now = 0.0
        self.free = [0.0, 0.0]
        self.end_at: list[float] = []
        self.timeline: list[dict] = []

    def _submit(self, fn) -> _Fut:
        self.tasks.append(fn)
        self.results.append(None)
        self.end_at.append(self.end_at[-1] if self.end_at else 0.0)
        return _TimedFut(self, len(self.tasks) - 1)

    def execute_model(self, so, non_block: bool = False):
        fut = super().execute_model(so, non_block)
        if so.total_num_scheduled_tokens:
            spec = so.scheduled_spec_decode_tokens
            rows = sum(1 for r, n in so.num_scheduled_tokens.items() if r in spec or (r.startswith("d") and n <= 6))
            pre = self.tok_scale * sum(n for r, n in so.num_scheduled_tokens.items() if r == "p")
            t, spans = self.now, []
            for s, (base, per_tok, per_row) in enumerate(self.cost):
                start = max(t, self.free[s])
                t = start + base + per_tok * pre + per_row * rows
                self.free[s] = t
                spans.append((start, t))
            self.end_at[fut.idx] = t
            self.timeline.append({"submit": self.now, "s0": spans[0], "s1": spans[1], "rows": rows, "pre": pre})
        else:
            self.end_at[fut.idx] = self.now
        return fut

    def sample_tokens(self, grammar_output, non_block: bool = False):
        fut = super().sample_tokens(grammar_output, non_block)
        self.end_at[fut.idx] = self.end_at[fut.idx - 1]
        return fut


def _hol(monkeypatch, lane, cost0, cost1, long_prompt=1792, chunk=128, scale=8):
    """Equivalent of a 14K prompt in 1024-token chunks (costs per real token, x `scale` per simulated token)."""
    ex = TimedPPExecutor(cost0, cost1, tok_scale=scale)
    ex, out, arrivals, info = _mixed_run(monkeypatch, lane=lane, executor=ex, long_prompt=long_prompt, chunk=chunk,
                                         max_tokens=400, max_model_len=2048)
    t0, t1 = info["t_inject"], info["t_first"]
    dec = [(t, n) for t, rid, n in arrivals if rid == "d0" and t0 < t <= t1]
    # verifies per second: tokens per verify are the drafter's acceptance (AL), independent of the lane
    verify_hz = len(dec) / (t1 - t0)
    gaps = [b - a for a, b in zip([t0] + [t for t, _ in dec], [t for t, _ in dec] + [t1])]
    return {"verify_hz": verify_hz, "prefill_tps": scale * long_prompt / (t1 - t0), "max_gap": max(gaps),
            "timeline": ex.timeline[info["step_inject"]:info["step_inject"] + 6]}


def _show(tag, r):
    print(f"\n{tag}: decoder {r['verify_hz']:.2f} verify/s (= {AL * r['verify_hz']:.1f} tok/s at AL {AL}), prefill {r['prefill_tps']:.0f} tok/s, "
          f"max gap {r['max_gap']:.2f} s")
    for e in r["timeline"]:
        print(f"  submit {e['submit']:7.3f}  s0 [{e['s0'][0]:7.3f},{e['s0'][1]:7.3f}]  "
              f"s1 [{e['s1'][0]:7.3f},{e['s1'][1]:7.3f}]  decode_rows={e['rows']} prefill={e['pre']}")


AL = 2.6   # accepted tokens per verify on bench prose (hol-design §1)
# Stage costs in seconds: (intercept, per prefill token, per decode row). Decode-only verify ~16 ms/stage.
COLD_ENGRAM = ((0.016, 1.9e-3, 0.0), (0.016, 0.12e-3, 0.0))      # stage 0 I/O bound (hol-design §1, random ids)
BALANCED = ((0.016, 0.33e-3, 0.0), (0.016, 0.33e-3, 0.0))        # compute bound, ~0.35 s/chunk on both stages


def test_timeline_stage0_bound_doubles_decoder(monkeypatch):
    off = _hol(monkeypatch, False, *COLD_ENGRAM)
    on = _hol(monkeypatch, True, *COLD_ENGRAM)
    _show("cold-Engram lane OFF", off)
    _show("cold-Engram lane ON ", on)
    assert on["verify_hz"] > 1.7 * off["verify_hz"]
    assert on["prefill_tps"] > 0.85 * off["prefill_tps"]          # stage 0 is the bottleneck either way
    assert on["max_gap"] < 0.7 * off["max_gap"]


def test_timeline_balanced_stages_trade_prefill(monkeypatch):
    off = _hol(monkeypatch, False, *BALANCED)
    on = _hol(monkeypatch, True, *BALANCED)
    _show("balanced lane OFF", off)
    _show("balanced lane ON ", on)
    # 2*max(s0,s1)/(s0+s1) = 1: the decoder gains little, the prefill loses its stage overlap
    assert 0.9 * off["verify_hz"] < on["verify_hz"] < 1.15 * off["verify_hz"]
    assert on["prefill_tps"] < 0.65 * off["prefill_tps"]
