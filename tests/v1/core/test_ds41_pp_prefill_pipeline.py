# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_DS41_PP_PREFILL_PIPELINE: pipelined chunked prefill under PP + sync scheduling + spec decode.

CPU only: a real Scheduler (PP=2, spec decode) driven by the real ``EngineCore.step_with_batch_queue`` /
``post_step`` against a fake executor whose futures resolve lazily and in FIFO order, like
``multiproc_executor.FutureWrapper`` (a result() drains every earlier future of the output rank).
"""

import queue
from collections import deque
from types import SimpleNamespace

import pytest

from vllm.v1.engine import ds41_pp_pipeline
from vllm.v1.engine.core import EngineCore
from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput

from .utils import create_requests, create_scheduler

KNOB = ds41_pp_pipeline.PP_PREFILL_PIPELINE_KNOB
NUM_SPEC = 2
BUDGET = 256  # max_num_batched_tokens (opt-125m caps max_model_len at 2048)


@pytest.fixture(autouse=True)
def _pp2_config_without_gpus(monkeypatch):
    """ParallelConfig refuses PP=2 when fewer devices are visible; these tests never touch a GPU."""
    from vllm.platforms import current_platform

    monkeypatch.setattr(current_platform, "device_count", lambda *a, **k: 8, raising=False)


# ----------------------------------------------------------------------------------------------- knob + predicate
def test_knob_default_off_and_strict(monkeypatch):
    monkeypatch.delenv(KNOB, raising=False)
    assert ds41_pp_pipeline.pp_prefill_pipeline_enabled() is False
    monkeypatch.setenv(KNOB, "1")
    assert ds41_pp_pipeline.pp_prefill_pipeline_enabled() is True
    monkeypatch.setenv(KNOB, "0")
    assert ds41_pp_pipeline.pp_prefill_pipeline_enabled() is False
    monkeypatch.setenv(KNOB, "maybe")
    with pytest.raises(ValueError):
        ds41_pp_pipeline.pp_prefill_pipeline_enabled()


def _so(num_scheduled, spec=None, **kw):
    return SimpleNamespace(
        num_scheduled_tokens=num_scheduled,
        total_num_scheduled_tokens=sum(num_scheduled.values()),
        scheduled_spec_decode_tokens=spec or {},
        **kw,
    )


def test_predicate_cases():
    part = SimpleNamespace(is_prefill_chunk=True)
    final = SimpleNamespace(is_prefill_chunk=False)
    reqs = {"a": part, "b": part, "c": final}
    f = ds41_pp_pipeline.is_partial_prefill_only
    assert f(_so({"a": 1024}), reqs)
    assert f(_so({"a": 512, "b": 512}), reqs)
    assert not f(_so({}), reqs)                                   # empty step
    assert not f(_so({"a": 1000, "c": 24}), reqs)                 # a final chunk / decode row in the step
    assert not f(_so({"a": 1024}, spec={"a": [1, 2]}), reqs)      # verify rows
    assert not f(_so({"a": 1024}, pending_structured_output_tokens=True), reqs)
    assert not f(_so({"a": 1024}, scheduled_ddtree_payloads={"a": object()}), reqs)
    assert not f(_so({"zz": 1024}), reqs)                         # unknown request


def test_predicate_on_real_scheduler():
    sched = create_scheduler(max_num_batched_tokens=BUDGET, pipeline_parallel_size=2,
                             num_speculative_tokens=NUM_SPEC)
    (req,) = create_requests(num_requests=1, num_tokens=600, ignore_eos=True)
    sched.add_request(req)
    flags = []
    for _ in range(3):  # PP: chunk k+1 is schedulable before chunk k's output is processed
        out = sched.schedule()
        assert out.num_scheduled_tokens[req.request_id] > 0
        flags.append(ds41_pp_pipeline.is_partial_prefill_only(out, sched.requests))
    assert flags == [True, True, False]  # 256 + 256 partial, the last 88 sample


# --------------------------------------------------------------------------------------------- engine-loop harness
class _Fut:
    def __init__(self, ex, kind, step):
        self.ex, self.kind, self.step = ex, kind, step
        self._done = False
        self._value = None
        ex.pending.append(self)

    def done(self):
        return self._done

    def _resolve(self):
        self._done = True
        if self.kind == "exec":
            self.ex.unresolved_exec.discard(self.step)
            # sync scheduling: execute_model returns None, except for an empty step
            self._value = self.ex.outputs[self.step] if self.step in self.ex.empty_steps else None
        else:
            self._value = self.ex.outputs[self.step]

    def result(self):
        while not self._done:
            self.ex.pending.popleft()._resolve()
        return self._value


class FakeExecutor:
    """Runner view per step: partial-prefill rows return no token; other rows accept every draft."""

    max_concurrent_batches = 2

    def __init__(self, scheduler):
        self.sched = scheduler
        self.pending = deque()
        self.unresolved_exec = set()
        self.outputs = {}
        self.empty_steps = set()
        self.step = -1
        self.last_req_ids = []
        self.inflight_at_submit = []
        self.draft_takes = 0

    def execute_model(self, so, non_block=True):
        self.step += 1
        # batches with GPU work still unresolved when this one is submitted
        self.inflight_at_submit.append(len(self.unresolved_exec))
        computed = {r.req_id: r.num_computed_tokens for r in so.scheduled_new_reqs}
        cached = so.scheduled_cached_reqs
        computed.update(zip(cached.req_ids, cached.num_computed_tokens))
        req_ids, sampled = [], []
        for rid, n in so.num_scheduled_tokens.items():
            req = self.sched.requests[rid]
            spec = so.scheduled_spec_decode_tokens.get(rid, [])
            end = computed[rid] + n - len(spec)
            req_ids.append(rid)
            if end < req.num_tokens:
                sampled.append([])                                  # partial prefill: discarded
            else:
                sampled.append([1000 + end + i for i in range(len(spec) + 1)])
        self.outputs[self.step] = ModelRunnerOutput(
            req_ids=req_ids, req_id_to_index={r: i for i, r in enumerate(req_ids)},
            sampled_token_ids=sampled)
        if not req_ids:
            self.empty_steps.add(self.step)
        else:
            self.last_req_ids = req_ids
        if req_ids:  # an empty step does no GPU work
            self.unresolved_exec.add(self.step)
        return _Fut(self, "exec", self.step)

    def sample_tokens(self, grammar_output, non_block=True):
        return _Fut(self, "sample", self.step)

    def take_draft_token_ids(self):
        # Blocking RPC to the output rank: drains every earlier future.
        self.draft_takes += 1
        while self.pending:
            self.pending.popleft()._resolve()
        return DraftTokenIds(req_ids=list(self.last_req_ids),
                             draft_token_ids=[[7] * NUM_SPEC for _ in self.last_req_ids])


def _make_core(sched, knob):
    core = object.__new__(EngineCore)
    core.scheduler = sched
    core.model_executor = FakeExecutor(sched)
    core.batch_queue_size = 2
    core.batch_queue = deque(maxlen=2)
    core.async_scheduling = False
    core.use_spec_decode = True
    core.is_ec_consumer = True
    core.is_pooling_model = False
    core._sm70_async_cpu_trace_step = 0
    core.aborts_queue = queue.Queue()
    core.vllm_config = SimpleNamespace(
        observability_config=SimpleNamespace(enable_logging_iteration_details=False))
    core._ds41_pp_prefill_pipeline = knob
    core._ds41_skip_draft_take = False
    return core


def _run(knob, prompt_lens, max_tokens=6):
    sched = create_scheduler(max_num_batched_tokens=BUDGET, pipeline_parallel_size=2,
                             num_speculative_tokens=NUM_SPEC)
    reqs = []
    for i, n in enumerate(prompt_lens):
        (r,) = create_requests(num_requests=1, num_tokens=n, ignore_eos=True, max_tokens=max_tokens,
                               req_ids=[f"r{i}"])
        reqs.append(r)
        sched.add_request(r)
    core = _make_core(sched, knob)
    schedule_log = []
    orig_schedule = sched.schedule

    def logged_schedule():
        out = orig_schedule()
        schedule_log.append((tuple(sorted(out.num_scheduled_tokens.items())),
                             tuple(sorted((k, tuple(v)) for k, v in out.scheduled_spec_decode_tokens.items()))))
        return out

    sched.schedule = logged_schedule
    for _ in range(500):
        if not (sched.has_requests() or core.batch_queue):
            break
        _, executed = core.step_with_batch_queue()
        core.post_step(executed)
    else:
        raise AssertionError("engine loop did not finish")
    outputs = {r.request_id: list(r.output_token_ids) for r in reqs}
    return schedule_log, outputs, core.model_executor


@pytest.mark.parametrize("prompt_lens", [[1500], [40, 1500], [700, 1100], [1500, 1500]])
def test_pipeline_same_steps_and_outputs(prompt_lens):
    log_off, out_off, ex_off = _run(False, prompt_lens)
    log_on, out_on, ex_on = _run(True, prompt_lens)
    # Knob off = today's behavior: the blocking draft take after every step -> one batch in flight.
    assert max(ex_off.inflight_at_submit) == 0
    # Same scheduler decisions, same tokens.
    assert log_on == log_off
    assert out_on == out_off
    for out in out_on.values():
        assert len(out) == 6
    # Knob on: a chunk enters while the previous partial chunk is in flight, never more than pp_size batches.
    assert max(ex_on.inflight_at_submit) == 1
    assert ex_on.draft_takes < ex_off.draft_takes


def test_pipeline_only_after_partial_prefill_steps():
    log_on, _, ex_on = _run(True, [40, 1500])
    for step, inflight in enumerate(ex_on.inflight_at_submit):
        if inflight and step > 0:
            prev_sched, prev_spec = log_on[step - 1]
            # the step in flight held no verify rows and only partial chunks of r1
            assert not prev_spec
            assert all(rid == "r1" for rid, _ in prev_sched)
