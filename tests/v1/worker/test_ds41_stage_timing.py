# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_DS41_STAGE_TIMING_EVERY: per-PP-stage step timing (CPU only, fake CUDA events on a fake clock)."""

from __future__ import annotations

import contextlib
from types import SimpleNamespace

import pytest

from vllm.sequence import IntermediateTensors
from vllm.v1.worker import ds41_stage_timing as st

pytestmark = pytest.mark.cpu_test


class Clock:
    def __init__(self):
        self.t = 0.0          # seconds

    def __call__(self):
        return self.t


class FakeEvent:
    """Records the clock when created; `ready` gates query() (an event the GPU has not reached yet)."""

    def __init__(self, clock, gate):
        self.t, self.gate = clock.t, gate

    def query(self):
        return self.gate["ready"]

    def elapsed_time(self, other):
        return (other.t - self.t) * 1000.0


def _so(sched, spec=None, new=()):
    return SimpleNamespace(num_scheduled_tokens=sched, total_num_scheduled_tokens=sum(sched.values()),
                           scheduled_spec_decode_tokens=spec or {},
                           scheduled_new_reqs=[SimpleNamespace(req_id=r) for r in new])


def _timer(every=2, is_last=False):
    clock, gate, lines = Clock(), {"ready": True}, []
    t = st.StageStepTimer(every=every, pp_rank=1 if is_last else 0, is_last=is_last,
                          log=lambda fmt, *a: lines.append(fmt % a),
                          event_factory=lambda: FakeEvent(clock, gate), clock=clock)
    return t, clock, gate, lines


def test_knob(monkeypatch):
    monkeypatch.delenv(st.STAGE_TIMING_KNOB, raising=False)
    assert st.stage_timing_every() == 0
    monkeypatch.setenv(st.STAGE_TIMING_KNOB, "50")
    assert st.stage_timing_every() == 50
    monkeypatch.setenv(st.STAGE_TIMING_KNOB, "-1")
    with pytest.raises(ValueError):
        st.stage_timing_every()


def test_classify():
    assert st.classify(_so({"d": 6}, spec={"d": [1] * 5})) == ("decode", 1, 6, 0)
    assert st.classify(_so({"d": 1})) == ("decode", 1, 1, 0)                      # zero-draft decode row
    assert st.classify(_so({"d": 6, "p": 1018}, spec={"d": [1] * 5})) == ("mixed", 1, 6, 1018)
    assert st.classify(_so({"p": 1024})) == ("prefill", 0, 0, 1024)
    assert st.classify(_so({"p": 1}, new=("p",))) == ("prefill", 0, 0, 1)       # one-token prompt


def test_stage0_steps_and_log_every_n():
    t, clock, gate, lines = _timer(every=2)
    # mixed step: 300 ms forward
    t.begin(_so({"d": 6, "p": 1018}, spec={"d": [1] * 5}))
    clock.t += 0.3
    t.fwd_done()
    t.finish()
    assert not lines
    # prefill-only step 0.5 s later, forward 250 ms; completes later (GPU not there yet)
    clock.t += 0.2
    gate["ready"] = False
    t.begin(_so({"p": 1024}))
    clock.t += 0.25
    t.finish()
    assert not lines                      # not polled until the events completed
    gate["ready"] = True
    t.begin(_so({}))                      # an empty step polls but is not timed
    assert len(lines) == 1
    line = lines[0]
    assert "pp_rank=0 steps=2" in line
    assert "mixed n=1 rows=1.0 dec_tok=6.0 pre_tok=1018 fwd=300.0ms" in line
    assert "prefill n=1 rows=0.0 dec_tok=0.0 pre_tok=1024 fwd=250.0ms" in line and "gap=500.0ms" in line
    assert "post=" not in line.split("|")[1]


def test_last_stage_recv_and_post():
    t, clock, gate, lines = _timer(every=1, is_last=True)
    t.begin(_so({"d": 6}, spec={"d": [1] * 5}))
    clock.t += 0.1                         # waiting for stage 0
    t.recv_done()
    clock.t += 0.016
    t.fwd_done()
    clock.t += 0.004                       # sampling + drafter + draft take
    t.finish()
    (line,) = lines
    assert "(last stage)" in line
    assert "decode n=1" in line and "fwd=16.0ms" in line and "post=4.0ms" in line and "span=120.0ms" in line


# --------------------------------------------------------------------------- wiring in Worker.execute_model
def _worker(monkeypatch, *, is_first, is_last, timer):
    from vllm.v1.worker import gpu_worker

    pp = SimpleNamespace(is_first_rank=is_first, is_last_rank=is_last, rank_in_group=0 if is_first else 1,
                         isend_tensor_dict=lambda *a, **k: [])
    monkeypatch.setattr(gpu_worker, "get_pp_group", lambda: pp)
    monkeypatch.setattr(gpu_worker, "get_tp_group", lambda: SimpleNamespace(rank_in_group=0))
    w = gpu_worker.Worker.__new__(gpu_worker.Worker)
    w._pp_send_work = []
    w._engram_service_cache = None
    w._ds41_pp_prefill_pipeline = False
    w._ds41_pp_fwd_done = None
    w._ds41_stage_timing_every = timer.every
    w._ds41_stage_timer = timer
    w.use_v2_model_runner = False
    w.vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(pass_config=SimpleNamespace(enable_sp=False)),
        parallel_config=SimpleNamespace(pipeline_parallel_size=2, distributed_executor_backend="mp"),
        scheduler_config=SimpleNamespace(async_scheduling=False))
    w.annotate_profile = lambda so: contextlib.nullcontext()
    w._schema_static_pp_send = lambda out, n: None
    w._use_sm70_static_pp_hidden_transfer = lambda n: False
    return w


def test_worker_wiring_first_stage(monkeypatch):
    import torch

    t, clock, gate, lines = _timer(every=1)
    w = _worker(monkeypatch, is_first=True, is_last=False, timer=t)

    def execute_model(so, it):
        clock.t += 0.2
        return IntermediateTensors({"h": torch.zeros(1)})

    w.model_runner = SimpleNamespace(execute_model=execute_model)
    assert w.execute_model(_so({"p": 1024})) is None
    (line,) = lines
    assert "pp_rank=0 steps=1" in line and "prefill n=1" in line and "fwd=200.0ms" in line


def test_worker_wiring_last_stage_finishes_in_sample_tokens(monkeypatch):
    t, clock, gate, lines = _timer(every=1, is_last=True)
    w = _worker(monkeypatch, is_first=True, is_last=True, timer=t)

    def execute_model(so, it):
        clock.t += 0.03
        return None                          # deferred sampling

    def sample_tokens(grammar_output):
        clock.t += 0.005
        return None

    w.model_runner = SimpleNamespace(execute_model=execute_model, sample_tokens=sample_tokens)
    assert w.execute_model(_so({"d": 6}, spec={"d": [1] * 5})) is None
    assert not lines
    w.sample_tokens(None)
    (line,) = lines
    assert "fwd=30.0ms" in line and "post=5.0ms" in line


def test_on_step_callback_without_logging():
    """Adaptive slice sizing builds the timer with every=0: no log lines, every completed step reported."""
    t, clock, gate, lines = _timer(every=0, is_last=True)
    seen = []
    t.on_step = lambda *a: seen.append(a)
    for slices in ((), ("x",)):
        so = _so({"d": 6}, spec={"d": [1] * 5})
        so.ds41_slices = slices
        t.begin(so)
        clock.t += 0.03
        t.fwd_done()
        clock.t += 0.007
        t.finish()
    assert not lines and len(seen) == 2
    kind, had, fwd, post, span = seen[0]
    assert kind == "decode" and had is False and fwd == pytest.approx(30.0) and post == pytest.approx(7.0)
    assert seen[1][1] is True
