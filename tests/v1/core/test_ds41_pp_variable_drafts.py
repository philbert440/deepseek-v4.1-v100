# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PP + synchronous scheduling + speculative decoding with a VARIABLE draft length per request and step.

The DeepSeek-V4.1 port serves DSpark on PP2 with the synchronous batch queue (``step_with_batch_queue`` + ``post_step``).
There the worker's drafts for step N are attached to the request (``post_step`` -> ``take_draft_token_ids``) BEFORE
``update_from_output`` has processed step N. The scheduler then has to wait for that output. With a fixed draft length
the wait was keyed on ``request.spec_token_ids`` being non-empty; a request whose drafter proposed ZERO drafts skipped
that guard and was scheduled with ``num_new_tokens = num_tokens - num_computed_tokens = -len(drafts of step N)``, a
negative count that reached the worker (``_prepare_inputs: assert total_num_scheduled_tokens > 0``, 2026-10-20).

This test drives the real ``Scheduler`` through the real ``EngineCore.step_with_batch_queue`` / ``post_step`` with a
FIFO fake executor (one worker queue, like the multiproc executor: an RPC completes every call queued before it) that
emulates both pipeline stages:

* the non-last stage's token bookkeeping from ``CachedRequestData.new_token_ids`` (``gpu_model_runner._update_states``
  after 7ac1eec1f1: num_new = num_computed + scheduled main tokens - tokens held), whose step input must equal the
  last stage's;
* the last stage's exact verification against a deterministic greedy "target" and a drafter whose drafts are right or
  wrong at random and truncated to a random length 0..k.

Gate: no negative / empty-but-submitted steps, every scheduled step carries its anchor (scheduled >= drafts + 1), both
stages see the same input ids, and every request's output equals the plain greedy continuation (spec decoding is exact).
"""

from __future__ import annotations

import dataclasses
import os
import queue
import random
from collections import deque

import pytest

from tests.v1.core.utils import create_requests, create_scheduler
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.engine.core import EngineCore
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, DraftTokenIds, ModelRunnerOutput
from vllm.v1.worker.gpu_model_runner import pp_stage_token_window

pytestmark = pytest.mark.cpu_test

K = 5
VOCAB = 1000


def target_next(seq: list[int]) -> int:
    """Deterministic greedy 'model': next token from the last two tokens and the length."""
    a = seq[-1] if seq else 0
    b = seq[-2] if len(seq) > 1 else 0
    return (a * 31 + b * 17 + len(seq) * 7 + 3) % VOCAB


def greedy_reference(prompt: list[int], n: int) -> list[int]:
    seq = list(prompt)
    for _ in range(n):
        seq.append(target_next(seq))
    return seq[len(prompt):]


class _Fut:
    def __init__(self, ex: "FakePPExecutor", idx: int):
        self.ex, self.idx = ex, idx

    def done(self) -> bool:
        return self.idx < self.ex.completed

    def result(self):
        self.ex.flush(self.idx + 1)
        return self.ex.results[self.idx]


class FakePPExecutor:
    """One FIFO worker queue covering both PP stages; ``take_draft_token_ids`` completes everything queued before it."""

    def __init__(self, length_policy, accuracy: float, seed: int):
        self.rng = random.Random(seed)
        self.length_policy = length_policy
        self.accuracy = accuracy
        self.tasks: list = []
        self.results: list = []
        self.completed = 0
        self.stage0: dict[str, list[int]] = {}   # non-last stage: prompt + output_token_ids (CachedRequestState)
        self.batch0: dict[str, dict] = {}         # non-last stage persistent batch rows: {"buf", "no_spec"}
        self.rpcs: list[str] = []                 # "exec" / "empty" / "sample" / "take" in submit order
        self.truth: dict[str, list[int]] = {}    # last stage: prompt + sampled tokens
        self.prompt_len: dict[str, int] = {}
        self.pending_output: ModelRunnerOutput | None = None
        self.last_drafts: DraftTokenIds | None = None
        self.steps: list[dict[str, tuple[int, int]]] = []   # per executed step: req -> (scheduled, drafts)

    # ---- executor API used by EngineCore ----
    def execute_model(self, scheduler_output: SchedulerOutput, non_block: bool = False):
        self.rpcs.append("exec" if scheduler_output.total_num_scheduled_tokens else "empty")
        return self._submit(lambda: self._execute(scheduler_output))

    def sample_tokens(self, grammar_output, non_block: bool = False):
        self.rpcs.append("sample")
        return self._submit(self._sample)

    def take_draft_token_ids(self) -> DraftTokenIds | None:
        self.rpcs.append("take")
        self.flush(len(self.tasks))
        return self.last_drafts

    # ---- FIFO machinery ----
    def _submit(self, fn) -> _Fut:
        self.tasks.append(fn)
        self.results.append(None)
        return _Fut(self, len(self.tasks) - 1)

    def flush(self, upto: int) -> None:
        while self.completed < upto:
            i = self.completed
            self.results[i] = self.tasks[i]()
            self.completed += 1

    # ---- the two stages ----
    def _execute(self, so: SchedulerOutput):
        total = so.total_num_scheduled_tokens
        # gpu_model_runner._update_states runs for every step, empty ones included: requests not scheduled in this
        # step leave the persistent batch (and are re-added from their cached state when scheduled again)
        for req_id in so.finished_req_ids:
            self.batch0.pop(req_id, None)
        for req_id in list(self.batch0):
            if req_id not in so.num_scheduled_tokens:
                del self.batch0[req_id]
        if not total:                       # gpu_model_runner.execute_model: empty step -> EMPTY output
            self.pending_output = None
            return EMPTY_MODEL_RUNNER_OUTPUT
        # gpu_model_runner._prepare_inputs (the 2026-10-20 crash site)
        assert total > 0, f"scheduled a step with total_num_scheduled_tokens={total}: {so.num_scheduled_tokens}"
        assert all(n > 0 for n in so.num_scheduled_tokens.values()), so.num_scheduled_tokens
        for new in so.scheduled_new_reqs:
            self.stage0[new.req_id] = list(new.prompt_token_ids)
            self.truth[new.req_id] = list(new.prompt_token_ids)
            self.prompt_len[new.req_id] = len(new.prompt_token_ids)
            self.batch0[new.req_id] = {"buf": list(new.prompt_token_ids), "no_spec": len(new.prompt_token_ids)}
        cached = so.scheduled_cached_reqs
        computed_of: dict[str, int] = {new.req_id: new.num_computed_tokens for new in so.scheduled_new_reqs}
        for i, req_id in enumerate(cached.req_ids):
            computed = cached.num_computed_tokens[i]
            computed_of[req_id] = computed
            spec = so.scheduled_spec_decode_tokens.get(req_id, ())
            num_main = so.num_scheduled_tokens.get(req_id, 0) - len(spec)
            held = self.stage0[req_id]
            new_ids = cached.new_token_ids[i]
            num_new = computed + num_main - len(held)
            assert num_new <= len(new_ids), (req_id, num_new, new_ids)
            if num_new > 0:
                held.extend(new_ids[-num_new:])
            num_output = cached.num_output_tokens[i]
            row = self.batch0.get(req_id)
            if num_output < len(held) - self.prompt_len[req_id]:
                del held[self.prompt_len[req_id] + num_output:]
                if row is not None:
                    row["no_spec"] = self.prompt_len[req_id] + num_output
            if row is None:                 # re-added from the cached state (input_batch.add_request)
                self.batch0[req_id] = {"buf": list(held), "no_spec": len(held)}
            else:                           # kept its row: the non-last-stage token window of _update_states
                start = row["no_spec"]
                end, toks = pp_stage_token_window(start, computed, num_main, list(new_ids))
                if end > start and new_ids:
                    _put(row["buf"], start, toks)
                row["no_spec"] = max(row["no_spec"], end)
        outputs: dict[str, list[int]] = {}
        drafts: dict[str, list[int]] = {}
        step_log: dict[str, tuple[int, int]] = {}
        for req_id, n in so.num_scheduled_tokens.items():
            spec = list(so.scheduled_spec_decode_tokens.get(req_id, ()))
            assert n >= len(spec) + 1, f"{req_id}: {n} scheduled for {len(spec)} drafts (no anchor)"
            step_log[req_id] = (n, len(spec))
            c = computed_of[req_id]
            num_main = n - len(spec)
            truth = self.truth[req_id]
            # both stages must feed the same ids: the non-last stage reads its persistent-batch row (token_ids_cpu)
            row = self.batch0[req_id]
            # (a prefill chunk's row holds the whole prompt; a decode row ends at its last main token)
            assert row["no_spec"] >= c + num_main and (not spec or row["no_spec"] == c + num_main), (
                req_id, row["no_spec"], c, num_main, len(spec))
            _put(row["buf"], row["no_spec"], spec)       # input_batch.update_req_spec_token_ids
            in0 = row["buf"][c:c + n]
            in1 = truth[c:c + num_main] + spec
            assert len(in1) == n and in0 == in1, (req_id, c, n, in0, in1)
            if c + num_main < len(truth):   # partial prefill chunk: nothing sampled
                outputs[req_id] = []
                continue
            assert c + num_main == len(truth), (req_id, c, num_main, len(truth))
            sampled = []
            for d in spec:
                exp = target_next(truth + sampled)
                if d != exp:
                    break
                sampled.append(d)
            sampled.append(target_next(truth + sampled))
            truth.extend(sampled)
            outputs[req_id] = sampled
            # drafter: k proposals continuing the accepted sequence, then a 0..k stop
            prop, seq = [], list(truth)
            for _ in range(K):
                tok = target_next(seq) if self.rng.random() < self.accuracy else self.rng.randrange(VOCAB)
                prop.append(tok)
                seq.append(tok)
            drafts[req_id] = prop[: self.length_policy(self.rng)]
        self.steps.append(step_log)
        req_ids = list(so.num_scheduled_tokens)
        self.pending_output = ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={r: i for i, r in enumerate(req_ids)},
            sampled_token_ids=[outputs[r] for r in req_ids],
        )
        self.last_drafts = DraftTokenIds(req_ids, [drafts.get(r, []) for r in req_ids])
        return None                          # sample_tokens() returns the output

    def _sample(self):
        out, self.pending_output = self.pending_output, None
        assert out is not None
        if os.environ.get("VLLM_DS41_FOLD_DRAFTS", "0") == "1":   # gpu_worker.sample_tokens
            out = dataclasses.replace(out, ds41_draft_token_ids=self.last_drafts)
        return out


def _put(buf: list[int], start: int, toks) -> None:
    toks = list(toks)
    if len(buf) < start + len(toks):
        buf.extend([-7] * (start + len(toks) - len(buf)))
    buf[start:start + len(toks)] = toks


def _core(scheduler, executor) -> EngineCore:
    core = EngineCore.__new__(EngineCore)
    core.scheduler = scheduler
    core.model_executor = executor
    core.vllm_config = scheduler.vllm_config
    core.batch_queue_size = 2
    core.batch_queue = deque(maxlen=2)
    core.is_ec_consumer = True
    core.is_pooling_model = False
    core.use_spec_decode = True
    core.async_scheduling = False
    core._ds41_pp_prefill_pipeline = False   # rivet/pp-pipeline attribute (set in __init__, which __new__ skips)
    core._ds41_skip_draft_take = False
    core.aborts_queue = queue.Queue()
    core._sm70_async_cpu_trace_step = 0
    return core


def _run(length_policy, *, num_requests: int, max_num_seqs: int, accuracy: float, seed: int,
         prompt_len: int = 37, max_tokens: int = 60, chunk: int = 8192, stagger: int = 0):
    scheduler = create_scheduler(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=chunk,
        max_model_len=4096,
        num_speculative_tokens=K,
        async_scheduling=False,
    )
    # ParallelConfig(pipeline_parallel_size=2) validates against the visible GPUs; the scheduler only reads the size
    # (use_pp: send sampled ids back to the non-last stage; the mixed-prefill budget is off under PP), so set it after.
    object.__setattr__(scheduler.parallel_config, "pipeline_parallel_size", 2)
    scheduler.use_pp = True
    executor = FakePPExecutor(length_policy, accuracy, seed)
    core = _core(scheduler, executor)
    reqs = create_requests(num_requests, num_tokens=prompt_len, max_tokens=max_tokens, ignore_eos=True)
    rng = random.Random(seed + 1)
    for r in reqs:      # distinct prompts
        r.prompt_token_ids[:] = [rng.randrange(VOCAB) for _ in r.prompt_token_ids]
        r._all_token_ids[:] = list(r.prompt_token_ids)
    pending = list(reqs)
    outputs: dict[str, list[int]] = {r.request_id: [] for r in reqs}
    iters = 0
    while pending or scheduler.has_requests() or core.batch_queue:
        iters += 1
        assert iters < 20000, "engine loop did not finish"
        # arrivals: one at a time (sequential prompts) or staggered
        if pending and scheduler.get_num_unfinished_requests() < max_num_seqs and iters % (stagger + 1) == 0:
            scheduler.add_request(pending.pop(0))
        if not scheduler.has_requests() and not core.batch_queue:
            continue
        engine_outputs, model_executed = core.step_with_batch_queue()
        core.post_step(model_executed)
        for eco in (engine_outputs or {}).values():
            for o in eco.outputs:
                outputs[o.request_id].extend(o.new_token_ids)
    for r in reqs:
        ref = greedy_reference(r.prompt_token_ids, max_tokens)
        assert outputs[r.request_id] == ref, (r.request_id, outputs[r.request_id][:20], ref[:20])
    executor.outputs = outputs
    executor.core = core
    return executor


FIXED = lambda rng: K                            # noqa: E731
RANDOM = lambda rng: rng.randrange(K + 1)        # noqa: E731
ZERO_HEAVY = lambda rng: 0 if rng.random() < 0.5 else rng.randrange(1, K + 1)   # noqa: E731
CAP3 = lambda rng: 3                             # noqa: E731


@pytest.mark.parametrize("seed", range(4))
def test_fixed_k_sequential_prompts(seed):
    _run(FIXED, num_requests=3, max_num_seqs=1, accuracy=0.8, seed=seed)


@pytest.mark.parametrize("policy", [RANDOM, ZERO_HEAVY, CAP3], ids=["random", "zero_heavy", "cap3"])
@pytest.mark.parametrize("seed", range(6))
def test_variable_drafts_b1_sequential_prompts(policy, seed):
    ex = _run(policy, num_requests=4, max_num_seqs=1, accuracy=0.8, seed=seed)
    if policy is CAP3:      # the cap path must really submit 3 drafts per verify step (not ~1)
        drafts = [d for step in ex.steps for (_, d) in step.values() if d]
        assert drafts and all(d == 3 for d in drafts)


@pytest.mark.parametrize("policy", [RANDOM, ZERO_HEAVY], ids=["random", "zero_heavy"])
@pytest.mark.parametrize("seed", range(6))
def test_variable_drafts_b2(policy, seed):
    _run(policy, num_requests=5, max_num_seqs=2, accuracy=0.7, seed=seed, stagger=3)


@pytest.mark.parametrize("seed", range(3))
def test_variable_drafts_chunked_prefill_b2(seed):
    # 300-token prompts in 64-token chunks: prefill chunks of one request interleave with the other's verify steps
    _run(RANDOM, num_requests=3, max_num_seqs=2, accuracy=0.8, seed=seed, prompt_len=300, chunk=64, stagger=2)


def test_rows_really_shrink():
    """With short drafts the verify steps carry fewer rows: scheduled == 1 + len(drafts) per request."""
    ex = _run(RANDOM, num_requests=2, max_num_seqs=1, accuracy=0.9, seed=11)
    lens = {d for step in ex.steps for (n, d) in step.values() if n == d + 1}
    assert len(lens) > 2      # several distinct row counts actually ran


# --------------------------------------------------------------- VLLM_DS41_PP_NO_EMPTY_STEP / VLLM_DS41_FOLD_DRAFTS
MODES = {
    "off": {},
    "no_empty": {"VLLM_DS41_PP_NO_EMPTY_STEP": "1"},
    "fold": {"VLLM_DS41_FOLD_DRAFTS": "1"},
    "both": {"VLLM_DS41_PP_NO_EMPTY_STEP": "1", "VLLM_DS41_FOLD_DRAFTS": "1"},
}


def _run_mode(monkeypatch, mode, policy, **kw):
    for k in ("VLLM_DS41_PP_NO_EMPTY_STEP", "VLLM_DS41_FOLD_DRAFTS"):
        monkeypatch.delenv(k, raising=False)
    for k, v in MODES[mode].items():
        monkeypatch.setenv(k, v)
    return _run(policy, **kw)


@pytest.mark.parametrize("mode", list(MODES))
@pytest.mark.parametrize("policy", [FIXED, RANDOM, ZERO_HEAVY], ids=["fixed", "random", "zero_heavy"])
@pytest.mark.parametrize("seed", range(3))
def test_no_empty_step_exact(monkeypatch, mode, policy, seed):
    """Every mode yields the plain greedy continuation (checked in _run) and the same tokens as the default,
    at B=1 (sequential prompts), B=2 staggered and with chunked prefill; the non-last stage keeps its
    persistent-batch row across steps with NO_EMPTY_STEP and must still feed the right ids."""
    for kw in (dict(num_requests=3, max_num_seqs=1, accuracy=0.8),
               dict(num_requests=4, max_num_seqs=2, accuracy=0.7, stagger=3),
               dict(num_requests=3, max_num_seqs=2, accuracy=0.8, prompt_len=300, chunk=64, stagger=2)):
        ref = _run_mode(monkeypatch, "off", policy, seed=seed, **kw).outputs
        got = _run_mode(monkeypatch, mode, policy, seed=seed, **kw).outputs
        assert got == ref


def _per_token_rpcs(monkeypatch, mode):
    ex = _run_mode(monkeypatch, mode, FIXED, num_requests=1, max_num_seqs=1, accuracy=0.8, seed=5, max_tokens=120)
    steps = sum(1 for r in ex.rpcs if r == "exec") - 1      # decode steps (minus the prefill)
    return {k: ex.rpcs.count(k) for k in ("exec", "empty", "sample", "take")}, steps


def test_b1_rpc_timeline(monkeypatch):
    """B=1 decode under the PP2 batch queue. Default: per verify step one real execute + one EMPTY execute
    + sample + take. NO_EMPTY_STEP drops the empty execute (one remains at the end: it carries the finished
    request id, which the workers must see); FOLD_DRAFTS drops the take."""
    counts, steps = _per_token_rpcs(monkeypatch, "off")
    assert counts["empty"] >= steps - 1 and counts["take"] >= steps
    counts, steps = _per_token_rpcs(monkeypatch, "no_empty")
    assert counts["empty"] <= 1 and counts["take"] >= steps
    counts, steps = _per_token_rpcs(monkeypatch, "fold")
    assert counts["take"] == 0 and counts["empty"] >= steps - 1
    counts, steps = _per_token_rpcs(monkeypatch, "both")
    assert counts["empty"] <= 1 and counts["take"] == 0
    assert counts["exec"] == counts["sample"] == steps + 1


def test_skippable_empty_keeps_side_effects():
    """An empty batch that carries finished ids, freed demand-KV blocks, ... is still submitted."""
    from vllm.v1.core.sched.output import CachedRequestData
    from vllm.v1.engine.core import ds41_skippable_empty

    def out(**kw):
        base = dict(scheduled_new_reqs=[], scheduled_cached_reqs=CachedRequestData.make_empty(),
                    num_scheduled_tokens={}, total_num_scheduled_tokens=0, scheduled_encoder_inputs={},
                    scheduled_spec_decode_tokens={}, num_common_prefix_blocks=[], finished_req_ids=set(),
                    free_encoder_mm_hashes=[])
        base.update(kw)
        return SchedulerOutput(**base)

    assert ds41_skippable_empty(out())
    assert not ds41_skippable_empty(out(finished_req_ids={"a"}))
    assert not ds41_skippable_empty(out(free_encoder_mm_hashes=["h"]))
    assert not ds41_skippable_empty(out(preempted_req_ids={"a"}))
    assert not ds41_skippable_empty(out(new_block_ids_to_zero=[3]))
    o = out()
    o.demand_release_block_ids = [7]
    assert not ds41_skippable_empty(o)


def test_pp_stage_token_window():
    """Kept persistent-batch row after a verify that accepted 2 drafts: 3 tokens (2 accepted + bonus) arrive, the
    stage needs exactly them (old formula num_computed + len(new) overshot by 2)."""
    from vllm.v1.worker.gpu_model_runner import pp_stage_token_window as w
    # stage holds 50 tokens, scheduler computed 52 (50 + 2 accepted), 1 main token, payload = tokens 50..52
    assert w(50, 52, 1, [7, 8, 9]) == (53, [7, 8, 9])
    # plain decode / prefill chunk: one main token, one id (unchanged behaviour)
    assert w(50, 50, 1, [7]) == (51, [7])
    assert w(100, 100, 64, list(range(64))) == (164, list(range(64)))
    # chunked prefill where the stage already holds more than num_computed: nothing to write
    assert w(120, 100, 8, list(range(8))) == (120, [])
    # async PP: no payload, advance to num_computed
    assert w(50, 53, 1, []) == (53, [])
