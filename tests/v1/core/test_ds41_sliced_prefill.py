# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_DS41_SLICED_PREFILL (hol-design.md Appendix A), scheduler/engine half. CPU only.

The real Scheduler runs through the real ``EngineCore.step_with_batch_queue`` / ``post_step`` against the FIFO
two-stage fake of test_ds41_pp_variable_drafts.py, extended with a worker-side model of the slice protocol:

* per-stage slice state keyed by (req_id, chunk_id): ``first`` creates it, slices must cover the stage's layers in
  order, ``last`` drops it; stage 0's ``last`` puts the handoff in this step's mailbox and stage 1's ``first`` must
  take it in the SAME step (every send has its receive); at most one chunk per stage; aborts drop state;
* per-stage KV: every normal step and every finished stage of a slice writes (position, token) into the request's
  blocks (worker block tables are rebuilt only from NewRequestData / CachedRequestData; slices use the directive's
  block_ids); every read (normal chunk, decode, a slice's prefix) checks that the KV it attends to is there;
* the last stage reports ``ds41_slice_done`` for a chunk whose stage-1 ``last`` slice ran.

Exactness: every request's output must equal the plain greedy continuation. The timeline test adds a simulated
clock with the costs measured on 8x V100-SXM2 (hol-design Appendix A) and prints decoder / prefill rates; run with -s.
"""

from __future__ import annotations

import dataclasses
import random
import types

import pytest

from tests.v1.core.test_ds41_pp_variable_drafts import (
    RANDOM,
    FakePPExecutor,
    _core,
    _Fut,
    greedy_reference,
)
from tests.v1.core.utils import create_requests, create_scheduler
from vllm.v1 import ds41_slices
from vllm.v1.core.sched import ds41_sliced_prefill
from vllm.v1.core.sched.ds41_sliced_prefill import AdaptiveSlicePolicy, SlicePlanner
from vllm.v1.ds41_slices import SliceDirective, plan_layer_slices
from vllm.v1.request import RequestStatus

pytestmark = pytest.mark.cpu_test

KNOBS = ("VLLM_DS41_DECODE_LANE", "VLLM_DS41_FOLD_DRAFTS", "VLLM_DS41_PP_NO_EMPTY_STEP", "VLLM_DS41_SLICED_PREFILL",
         "VLLM_DS41_SLICE_LAYERS_S0", "VLLM_DS41_SLICE_LAYERS_S1")
S0, S1 = (0, 20), (20, 40)
BLOCK = 16


# ------------------------------------------------------------------------------------------- shared module / knobs
def test_knobs_and_layer_plans(monkeypatch):
    for k in KNOBS:
        monkeypatch.delenv(k, raising=False)
    assert ds41_slices.sliced_prefill_enabled() is False
    assert ds41_slices.slice_layers_s0() == 3 and ds41_slices.slice_layers_s1() == 0
    monkeypatch.setenv("VLLM_DS41_SLICED_PREFILL", "1")
    assert ds41_slices.sliced_prefill_enabled() is True
    monkeypatch.setenv("VLLM_DS41_SLICED_PREFILL", "perhaps")
    with pytest.raises(ValueError):
        ds41_slices.sliced_prefill_enabled()
    assert plan_layer_slices(0, 20, 3) == ((0, 3), (3, 6), (6, 9), (9, 12), (12, 15), (15, 18), (18, 20))
    assert plan_layer_slices(20, 40, 0) == ((20, 21), (21, 40))
    assert plan_layer_slices(20, 21, 0) == ((20, 21),)
    assert plan_layer_slices(20, 40, 10) == ((20, 30), (30, 40))


def test_contract_fields_default_empty():
    import pickle

    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput

    so = SchedulerOutput.make_empty()
    assert so.ds41_slices == () and so.ds41_slice_aborts == () and EMPTY_MODEL_RUNNER_OUTPUT.ds41_slice_done == ()
    d = SliceDirective(0, "r", 3, 64, 32, tuple(range(32)), ((1, 2, 3, 4, 5, 6),), 0, 3, True, False, 500)
    so.ds41_slices, so.ds41_slice_aborts = (d,), (("q", 1),)
    so2 = pickle.loads(pickle.dumps(so))
    assert so2.ds41_slices == (d,) and so2.ds41_slice_aborts == (("q", 1),) and d.key == ("r", 3) and d.end_pos == 96
    out = ModelRunnerOutput(req_ids=[], req_id_to_index={}, ds41_slice_done=(("r", 3),))
    assert pickle.loads(pickle.dumps(out)).ds41_slice_done == (("r", 3),)


def test_engine_requires_fold_drafts():
    import inspect

    from vllm.v1.engine import ds41_decode_lane
    from vllm.v1.engine.core import EngineCore

    with pytest.raises(ValueError, match="VLLM_DS41_SLICED_PREFILL=1 requires VLLM_DS41_FOLD_DRAFTS=1"):
        ds41_decode_lane.check_config(fold_drafts=False, async_scheduling=False, knob="VLLM_DS41_SLICED_PREFILL")
    src = inspect.getsource(EngineCore.__init__)
    assert "knob=ds41_slices.SLICED_PREFILL_KNOB" in src and "self._ds41_decode_lane = True" in src


# ------------------------------------------------------------------------------------------- worker-side fake
class SlicedPPExecutor(FakePPExecutor):
    """FakePPExecutor + the Appendix-A worker protocol (see module docstring)."""

    def __init__(self, policy=RANDOM, accuracy=0.8, seed=0, *, s0=S0, s1=S1):
        super().__init__(policy, accuracy, seed)
        self.ranges = (s0, s1)
        self.seq: dict[str, list[int]] = {}           # every request's prompt (+ sampled), registered by the test
        self.tables: dict[str, list[int]] = {}        # worker block tables (KV group 0), from scheduler messages
        self.kv = ({}, {})                            # per stage: block_id -> {slot: (pos, token)}
        self.state = ({}, {})                         # per stage: (req_id, chunk_id) -> next layer
        self.parts = ({}, {})                         # per stage: key -> next attention part of the layer, or "moe"
        self.done_end: dict[str, int] = {}
        self.slice_steps: list[tuple[SliceDirective, ...]] = []
        self.normal_steps: list[dict[str, int]] = []
        self.aborts: list[tuple[str, int]] = []
        self.restart: set[str] = set()                # requests whose chunks were aborted since their last done
        self.so_log: list = []
        self.part_log: list = []

    # ---- KV model
    def _write(self, stage, table, start, toks):
        for i, tok in enumerate(toks):
            pos = start + i
            self.kv[stage].setdefault(table[pos // BLOCK], {})[pos % BLOCK] = (pos, tok)

    def _check(self, stage, table, upto, rid, what):
        seq = self.seq[rid]
        for pos in range(upto):
            got = self.kv[stage].get(table[pos // BLOCK], {}).get(pos % BLOCK)
            assert got == (pos, seq[pos]), f"{what}: {rid} stage {stage} pos {pos}: KV {got} != {(pos, seq[pos])}"

    def _track_tables(self, so):
        for rid in so.finished_req_ids:
            self.tables.pop(rid, None)
        for new in so.scheduled_new_reqs:
            self.tables[new.req_id] = list(new.block_ids[0])
        cached = so.scheduled_cached_reqs
        for i, rid in enumerate(cached.req_ids):
            ids = cached.new_block_ids[i]
            if rid in cached.resumed_req_ids:
                self.tables[rid] = list(ids[0]) if ids is not None else []
            elif ids is not None:
                self.tables[rid].extend(ids[0])

    def _execute(self, so):
        self.so_log.append(so)
        for key in so.ds41_slice_aborts:          # drop per-chunk state, free nothing
            self.aborts.append(tuple(key))
            self.restart.add(key[0])
            for st in (*self.state, *self.parts):
                st.pop(tuple(key), None)
        self._track_tables(so)
        if so.ds41_slices:
            assert so.total_num_scheduled_tokens > 0, "slices may only ride on a step that carries decode rows"
        # normal rows: check the attended prefix, write this step's tokens (main + drafts) on both stages
        computed = {new.req_id: new.num_computed_tokens for new in so.scheduled_new_reqs}
        cached = so.scheduled_cached_reqs
        computed.update(zip(cached.req_ids, cached.num_computed_tokens))
        log = {}
        for rid, n in so.num_scheduled_tokens.items():
            spec = list(so.scheduled_spec_decode_tokens.get(rid, ()))
            c, main = computed[rid], n - len(spec)
            table = self.tables[rid]
            assert len(table) * BLOCK >= c + n, (rid, len(table), c, n)
            for stage in (0, 1):
                self._check(stage, table, c, rid, "normal step")
                self._write(stage, table, c, self.seq[rid][c:c + main] + spec)
            log[rid] = n
        self.normal_steps.append(log)
        result = super()._execute(so)
        if self.pending_output is not None:
            for rid, toks in zip(self.pending_output.req_ids, self.pending_output.sampled_token_ids):
                if toks:
                    self.seq[rid] = self.truth[rid][:]
        done = self._run_slices(so.ds41_slices)
        if done:
            assert self.pending_output is not None
            self.pending_output = dataclasses.replace(self.pending_output, ds41_slice_done=tuple(done))
        return result

    def _run_slices(self, slices):
        self.slice_steps.append(tuple(slices))
        mailbox: list[tuple[str, int]] = []
        done = []
        for stage in (0, 1):                        # stage 0 never waits on stage 1: run it first
            a_stage, b_stage = self.ranges[stage]
            for d in (x for x in slices if x.stage == stage):
                key = d.key
                st = self.state[stage]
                assert d.layer_start < d.layer_end and a_stage <= d.layer_start and d.layer_end <= b_stage, d
                if d.first:
                    assert key not in st and d.layer_start == a_stage, d
                    if stage == 1:
                        assert mailbox and mailbox.pop(0) == key, f"stage-1 first {key} without a same-step handoff"
                    st[key] = a_stage
                    seq = self.seq[d.req_id]
                    if stage == 0:
                        assert d.token_ids == tuple(seq[d.start_pos:d.end_pos]), "stage-0 token ids"
                    else:
                        assert d.token_ids == ()
                    table = [b for b in d.block_ids[0]]
                    assert len(table) == -(-d.end_pos // BLOCK), (len(table), d.end_pos)
                    worker = self.tables.get(d.req_id)
                    if worker is not None:                 # the workers' own table agrees on the shared prefix
                        n = min(len(worker), len(table))
                        assert worker[:n] == table[:n], (worker[:n], table[:n])
                    self._check(stage, table, d.start_pos, d.req_id, "slice prefix")
                    assert d.n_req == len(self.prompt_of(d.req_id)), d
                assert st.get(key) == d.layer_start, f"stage {stage} slice {d} out of order (at {st.get(key)})"
                parts = self.parts[stage]
                if d.layer_part == 1:              # attention part i/n of layer_start: the layer is not done yet
                    assert stage == 1 and d.layer_end == d.layer_start + 1 and not d.last, d
                    assert parts.get(key, 0) == d.part_index < d.part_count, (d, parts.get(key))
                    parts[key] = "moe" if d.part_index + 1 == d.part_count else d.part_index + 1
                    self.part_log.append((d.key, d.part_index, d.part_count))
                    continue
                if d.layer_part == 2:              # MoE half of layer_start, then whole layers
                    assert parts.pop(key, None) == "moe", (d, "MoE part before every attention part")
                else:
                    assert key not in parts, (d, "whole-layer slice inside a split layer")
                st[key] = d.layer_end
                if d.last:
                    assert d.layer_end == b_stage
                    del st[key]
                    self._write(stage, list(d.block_ids[0]), d.start_pos,
                                list(self.seq[d.req_id][d.start_pos:d.end_pos]))
                    if stage == 0:
                        mailbox.append(key)
                    else:
                        done.append(key)
                        if d.req_id not in self.restart:
                            assert self.done_end.get(d.req_id, d.start_pos) == d.start_pos, "chunks complete in order"
                        self.restart.discard(d.req_id)
                        self.done_end[d.req_id] = d.end_pos
                assert len(st) <= 1, f"more than one chunk in flight on stage {stage}: {st}"
        assert not mailbox, f"handoff sent without a receive in the same step: {mailbox}"
        return done

    def prompt_of(self, rid):
        return self.prompts[rid]


# ------------------------------------------------------------------------------------------- driver
def _setup(monkeypatch, *, sliced, lane=None, chunk=256, max_model_len=4096, decoders=1, prefix_caching=False,
           min_tail=32, s0_per=3, s1_per=0, long_prefill=0, policy=None, attn_parts=None):
    for k in KNOBS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("VLLM_DS41_FOLD_DRAFTS", "1")
    if sliced:
        monkeypatch.setenv("VLLM_DS41_SLICED_PREFILL", "1")
    sched = create_scheduler(max_num_seqs=decoders + 2, max_num_batched_tokens=chunk, max_model_len=max_model_len,
                             num_speculative_tokens=5, async_scheduling=False, enable_prefix_caching=prefix_caching,
                             block_size=BLOCK, long_prefill_token_threshold=long_prefill)
    object.__setattr__(sched.parallel_config, "pipeline_parallel_size", 2)
    sched.use_pp = True
    if sliced:
        assert sched._ds41_slicer is not None
        sched._ds41_slicer = SlicePlanner(S0, S1, chunk, s0_per_slice=s0_per, s1_per_slice=s1_per, min_tail=min_tail,
                                          policy=policy, s1_attn_parts=attn_parts)
    else:
        assert sched._ds41_slicer is None
    return sched


def _engine(sched, ex, *, sliced, lane):
    core = _core(sched, ex)
    core._ds41_decode_lane = sliced or lane
    core._ds41_sliced_prefill = sliced
    core._ds41_lane_holds = 0
    return core


def run(monkeypatch, *, sliced=True, lane=False, decoders=1, long_prompt=1500, chunk=256, inject_after=8,
        dec_max_tokens=200, seed=0, executor=None, events=(), extra=(), check=None, max_model_len=4096,
        prefix_caching=False, min_tail=32, s0_per=3, s1_per=0, same_prompt=False, policy=None, attn_parts=None):
    """``decoders`` short requests start decoding; once d0 has ``inject_after`` tokens the long prompt "p" (and any
    ``extra`` (req_id, prompt_len, kwargs)) arrives. ``events``: [(predicate(state), action(state))] fired once."""
    sched = _setup(monkeypatch, sliced=sliced, chunk=chunk, max_model_len=max_model_len, decoders=decoders + len(extra),
                   prefix_caching=prefix_caching, min_tail=min_tail, s0_per=s0_per, s1_per=s1_per, policy=policy,
                   attn_parts=attn_parts)
    ex = executor if executor is not None else SlicedPPExecutor(RANDOM, 0.8, seed)
    ex.prompts = {}
    core = _engine(sched, ex, sliced=sliced, lane=lane)
    rng = random.Random(seed + 1)
    short = create_requests(decoders, num_tokens=37, max_tokens=dec_max_tokens, ignore_eos=True,
                            req_ids=[f"d{i}" for i in range(decoders)], block_size=BLOCK)
    (big,) = create_requests(1, num_tokens=long_prompt, max_tokens=4, ignore_eos=True, req_ids=["p"], block_size=BLOCK)
    others = [create_requests(1, num_tokens=n, ignore_eos=True, req_ids=[rid], block_size=BLOCK, **kw)[0]
              for rid, n, kw in extra]
    if not same_prompt:
        for r in (*short, big, *others):
            r.prompt_token_ids[:] = [rng.randrange(1000) for _ in r.prompt_token_ids]
            r._all_token_ids[:] = list(r.prompt_token_ids)
    for r in (*short, big, *others):
        ex.prompts[r.request_id] = list(r.prompt_token_ids)
        ex.seq[r.request_id] = list(r.prompt_token_ids)
    for r in short:
        sched.add_request(r)
    outputs = {r.request_id: [] for r in (*short, big, *others)}
    st = {"sched": sched, "core": core, "ex": ex, "big": big, "outputs": outputs, "fired": set(), "info": {},
          "aborted": set()}
    for _ in range(40000):
        if not (sched.has_requests() or core.batch_queue):
            break
        if "t_inject" not in st["info"] and len(outputs["d0"]) >= inject_after:
            for r in (big, *others):
                sched.add_request(r)
            st["info"]["t_inject"] = getattr(ex, "now", 0.0)
            st["info"]["step_inject"] = len(ex.steps)
        for i, (pred, action) in enumerate(events):
            if i not in st["fired"] and pred(st):
                st["fired"].add(i)
                action(st)
        eco_map, executed = core.step_with_batch_queue()
        core.post_step(executed)
        for eco in (eco_map or {}).values():
            for o in eco.outputs:
                outputs[o.request_id].extend(o.new_token_ids)
                if o.new_token_ids:
                    st["info"].setdefault("arrivals", []).append((getattr(ex, "now", 0.0), o.request_id))
                    if o.request_id == "p" and "t_first" not in st["info"]:
                        st["info"]["t_first"] = getattr(ex, "now", 0.0)
        if check is not None:
            check(st)
        _invariants(st)
    else:
        raise AssertionError("engine loop did not finish")
    for r in (*short, big, *others):
        if r.request_id in st["aborted"]:
            continue
        assert outputs[r.request_id] == greedy_reference(r.prompt_token_ids, r.max_tokens), r.request_id
    # everything freed
    pool = sched.kv_cache_manager.block_pool
    assert pool.get_num_free_blocks() == pool.num_gpu_blocks - 1
    slicer = sched._ds41_slicer
    if slicer is not None:
        assert not slicer.in_flight and not slicer.awaiting and slicer.s0 is None and slicer.s1 is None
        assert not slicer.unseen and not slicer.pending_blocks
    return st


def _invariants(st):
    sched, ex, slicer = st["sched"], st["ex"], st["sched"]._ds41_slicer
    if slicer is None:
        return
    for rid, q in slicer.in_flight.items():
        req = sched.requests[rid]
        # num_computed_tokens advances only on slice_done (the fake's done tally == scheduler's view)
        assert req.num_computed_tokens == q[0].start, (rid, req.num_computed_tokens, q[0].start)
        if rid not in ex.restart:          # (the workers may have finished q[0] before its done is reported)
            assert ex.done_end.get(rid, q[0].start) <= q[0].end
        # staged blocks are not hashed until slice_done
        mgr = sched.kv_cache_manager.coordinator.single_type_managers[0]
        blocks = mgr.req_to_blocks.get(rid, [])
        for b in blocks[q[0].start // BLOCK:]:
            assert b.block_hash is None, f"staged block {b.block_id} of {rid} hashed before slice_done"
        # never a final chunk; at least min_tail tokens stay for the normal path
        assert req.num_tokens - q[-1].end >= slicer.min_tail
        assert not req.spec_token_ids


def _staged_while_decoding(st):
    ex = st["ex"]
    return [s for s in ex.slice_steps if s]


# ------------------------------------------------------------------------------------------- behaviour
@pytest.mark.parametrize("seed", range(3))
def test_staging_advance_and_final_chunk_normal(monkeypatch, seed):
    st = run(monkeypatch, seed=seed)
    ex, big = st["ex"], st["big"]
    # every non-final chunk of "p" ran as slices, the final chunk (>= min_tail) ran on the normal path ONCE
    p_steps = [s["p"] for s in ex.normal_steps if "p" in s and s["p"] > 6]     # (later p steps are verifies)
    planner = st["sched"]._ds41_slicer
    start, expect = 0, []
    while planner.chunk_len(big, start):
        expect.append(planner.chunk_len(big, start))
        start += expect[-1]
    assert expect == [256] * 5 and p_steps == [1500 - 1280], p_steps        # 5 staged chunks + a 220-token tail
    assert ex.done_end["p"] + p_steps[0] == big.num_prompt_tokens
    chunks = sorted({d.key for s in ex.slice_steps for d in s})
    assert [c for _, c in chunks] == list(range(len(chunks))) and len(chunks) == 5
    # every slice step carried decode rows; stage 0 ran 7 slices per chunk, stage 1 two
    for s in ex.slice_steps:
        for d in s:
            assert d.stage in (0, 1)
    per_stage = [sum(1 for s in ex.slice_steps for d in s if d.stage == k) for k in (0, 1)]
    assert per_stage == [7 * len(chunks), 2 * len(chunks)]
    # pipelining: stage 0 started chunk k+1 while stage 1 still had chunk k
    overlap = any({(d.stage, d.chunk_id) for d in s} >= {(0, k + 1), (1, k)} for s in ex.slice_steps for k in range(4))
    assert overlap
    # the long prompt reached the workers as a NEW request at its final chunk, with the staged prefix computed
    news = [n for so in ex.so_log for n in so.scheduled_new_reqs if n.req_id == "p"]
    assert len(news) == 1 and news[0].num_computed_tokens == ex.done_end["p"]
    assert st["core"]._ds41_lane_holds > 0 and ex.rpcs.count("take") == 0


def test_running_prompt_staged_mid_prefill(monkeypatch):
    """The long prompt started BEFORE any decoder (normal pipelined chunks); a short request arrives, decodes, and
    the rest of the prompt is staged. The workers hold "p": the staged blocks are appended to its new_block_ids."""
    for k in KNOBS:
        monkeypatch.delenv(k, raising=False)
    # long_prefill_token_threshold leaves budget room so the short request is admitted during the normal prefill
    sched = _setup(monkeypatch, sliced=True, chunk=128, decoders=1, long_prefill=96)
    ex = SlicedPPExecutor(RANDOM, 0.8, 3)
    core = _engine(sched, ex, sliced=True, lane=False)
    rng = random.Random(9)
    (big,) = create_requests(1, num_tokens=1600, max_tokens=3, ignore_eos=True, req_ids=["p"], block_size=BLOCK)
    (dec,) = create_requests(1, num_tokens=20, max_tokens=120, ignore_eos=True, req_ids=["d0"], block_size=BLOCK)
    ex.prompts = {}
    for r in (big, dec):
        r.prompt_token_ids[:] = [rng.randrange(1000) for _ in r.prompt_token_ids]
        r._all_token_ids[:] = list(r.prompt_token_ids)
        ex.prompts[r.request_id] = list(r.prompt_token_ids)
        ex.seq[r.request_id] = list(r.prompt_token_ids)
    sched.add_request(big)
    out = {"p": [], "d0": []}
    for i in range(20000):
        if not (sched.has_requests() or core.batch_queue):
            break
        if i == 4:
            sched.add_request(dec)
        eco_map, executed = core.step_with_batch_queue()
        core.post_step(executed)
        for eco in (eco_map or {}).values():
            for o in eco.outputs:
                out[o.request_id].extend(o.new_token_ids)
        _invariants({"sched": sched, "ex": ex})
    assert out["p"] == greedy_reference(big.prompt_token_ids, 3)
    assert out["d0"] == greedy_reference(dec.prompt_token_ids, 120)
    normal_p = [s["p"] for s in ex.normal_steps if "p" in s]
    assert len(normal_p) >= 3 and ex.done_end["p"] > 0          # normal chunks first, staged after, normal tail
    news = [n for so in ex.so_log for n in so.scheduled_new_reqs if n.req_id == "p"]
    assert len(news) == 1 and news[0].num_computed_tokens == 0  # sent once, as a normal new request


def test_two_decoders(monkeypatch):
    st = run(monkeypatch, decoders=2, seed=5)
    ex = st["ex"]
    assert ex.done_end["p"] > 0
    # slices rode on steps with decode rows only (asserted in the fake); both decoders kept decoding during it
    steps_with_slices = [i for i, s in enumerate(ex.slice_steps) if s]
    rows = {r for i in steps_with_slices for r in ex.normal_steps[i]}
    assert {"d0", "d1"} <= rows


def test_no_decoder_uses_normal_path(monkeypatch):
    """A lone long prompt: no directives at all, the same steps and RPCs as the Phase-0 lane engine (sliced mode
    always runs the lane hold; with no decoder it never holds until the final chunk is in flight)."""
    def lone(sliced):
        sched = _setup(monkeypatch, sliced=sliced, chunk=128, decoders=0)
        ex = SlicedPPExecutor(RANDOM, 0.8, 0)
        core = _engine(sched, ex, sliced=sliced, lane=True)
        (r,) = create_requests(1, num_tokens=1000, max_tokens=5, ignore_eos=True, req_ids=["p"], block_size=BLOCK)
        ex.prompts = {"p": list(r.prompt_token_ids)}
        ex.seq["p"] = list(r.prompt_token_ids)
        sched.add_request(r)
        out = []
        for _ in range(500):
            if not (sched.has_requests() or core.batch_queue):
                break
            eco_map, executed = core.step_with_batch_queue()
            core.post_step(executed)
            for eco in (eco_map or {}).values():
                for o in eco.outputs:
                    out.extend(o.new_token_ids)
        assert out == greedy_reference(r.prompt_token_ids, 5)
        return ex
    on, off = lone(True), lone(False)
    assert not any(on.slice_steps) and (on.rpcs, on.steps) == (off.rpcs, off.steps)


def test_knob_off_identical(monkeypatch):
    """Knob off: no planner, every SchedulerOutput carries empty slice fields, and the step sequence equals the
    decode-lane-only engine (Phase 0) on the same workload; knob on changes the schedule only by staging."""
    off = run(monkeypatch, sliced=False, lane=True, seed=2)
    off2 = run(monkeypatch, sliced=False, lane=True, seed=2)
    assert all(so.ds41_slices == () and so.ds41_slice_aborts == () for so in off["ex"].so_log)
    assert (off["ex"].rpcs, off["ex"].steps) == (off2["ex"].rpcs, off2["ex"].steps)
    plain = run(monkeypatch, sliced=False, lane=False, seed=2)
    assert all(so.ds41_slices == () for so in plain["ex"].so_log)
    # knob on with nothing to stage (the "long" prompt fits one chunk): exactly the Phase-0 lane schedule
    short_on = run(monkeypatch, sliced=True, long_prompt=200, seed=2)
    short_lane = run(monkeypatch, sliced=False, lane=True, long_prompt=200, seed=2)
    assert not any(short_on["ex"].slice_steps)
    assert (short_on["ex"].rpcs, short_on["ex"].steps) == (short_lane["ex"].rpcs, short_lane["ex"].steps)


def test_abort_mid_chunk(monkeypatch):
    """The client aborts "p" with two chunks in flight (one on stage 0, one whose stage-1 slices are submitted):
    the next step carries ds41_slice_aborts for both, nothing of "p" runs afterwards, the late slice_done is ignored
    and its blocks are freed (checked at the end of run())."""
    def staged_on_both(st):
        s = st["sched"]._ds41_slicer
        return s.s0 is not None and len(s.in_flight.get("p", ())) >= 2

    def abort(st):
        s = st["sched"]._ds41_slicer
        st["info"]["abort_keys"] = sorted(c.key for q in s.in_flight.values() for c in q)
        st["sched"].finish_requests("p", RequestStatus.FINISHED_ABORTED)
        st["aborted"].add("p")

    st = run(monkeypatch, events=[(staged_on_both, abort)], seed=1)
    ex, info = st["ex"], st["info"]
    assert info["abort_keys"] and sorted(ex.aborts) == info["abort_keys"]
    first = next(i for i, so in enumerate(ex.so_log) if so.ds41_slice_aborts)
    after = ex.so_log[first:]
    assert not any(d.req_id == "p" for so in after for d in so.ds41_slices)
    assert not any("p" in so.num_scheduled_tokens for so in after)
    assert st["outputs"]["p"] == []


def test_decoder_finishes_while_staged(monkeypatch):
    """The only decoder finishes mid-prompt: chunks still on a stage are aborted and the prompt continues on the
    normal path from the last committed chunk (exactness checked in run())."""
    st = run(monkeypatch, dec_max_tokens=40, inject_after=8, long_prompt=1900, seed=4)
    ex, slicer = st["ex"], st["sched"]._ds41_slicer
    assert ex.done_end.get("p", 0) > 0 and ex.aborts and slicer.stats.by_reason.get("no_decoder", 0) > 0
    p_normal = [s["p"] for s in ex.normal_steps if "p" in s]
    assert len(p_normal) >= 2                                    # several normal chunks after the fallback


def test_preempted_while_staged(monkeypatch):
    """KV pressure: the decoder's allocation preempts the staged prompt (last in running). Its chunks are aborted,
    the unseen request restarts as a NEW request, and everything is still exact."""
    def staged(st):
        return st["sched"]._ds41_slicer.s0 is not None and st["ex"].done_end.get("p", 0) > 0

    def squeeze(st):
        sched = st["sched"]
        pool = sched.kv_cache_manager.block_pool
        st["info"]["held"] = pool.get_new_blocks(pool.get_num_free_blocks())   # leave no free blocks

    def release(st):
        return "held" in st["info"] and st["sched"]._ds41_slicer.stats.by_reason.get("preempt", 0) > 0

    def give_back(st):
        st["sched"].kv_cache_manager.block_pool.free_blocks(st["info"].pop("held"))

    st = run(monkeypatch, events=[(staged, squeeze), (release, give_back)], seed=6, dec_max_tokens=160)
    slicer, ex = st["sched"]._ds41_slicer, st["ex"]
    assert slicer.stats.by_reason.get("preempt", 0) > 0 and ex.aborts
    news = [n for so in ex.so_log for n in so.scheduled_new_reqs if n.req_id == "p"]
    assert len(news) == 1


def test_prompt_logprobs_and_short_prompts_use_normal_path(monkeypatch):
    """A prompt-logprob request never gets staged; a short prompt arriving while "p" is staged is admitted
    normally (its only chunk is final)."""
    planner = SlicePlanner(S0, S1, 256, min_tail=32)
    (lp,) = create_requests(1, num_tokens=1200, prompt_logprobs=1, req_ids=["lp"], block_size=BLOCK)
    assert not planner.eligible(lp) and not planner.defer_waiting(lp, True)
    st = run(monkeypatch, extra=[("s", 60, {"max_tokens": 6})], seed=7)
    ex = st["ex"]
    assert not any(d.req_id == "s" for s in ex.slice_steps for d in s)
    s_first = next(i for i, s in enumerate(ex.normal_steps) if "s" in s)
    p_final = next(i for i, s in enumerate(ex.normal_steps) if "p" in s)
    assert s_first < p_final                     # admitted while "p" was still staged


def test_prefix_cache_hit_and_late_hash(monkeypatch):
    """Prefix caching on: blocks of a staged chunk are not hashed until its slice_done (checked every step in
    _invariants); after "p" finishes, an identical prompt hits the cache for the staged prefix."""
    st = run(monkeypatch, prefix_caching=True, same_prompt=True, long_prompt=1200, extra=[("q", 1200, {})], seed=8)
    ex = st["ex"]
    q_starts = [d.start_pos for s in ex.slice_steps for d in s if d.req_id == "q" and d.first and d.stage == 0]
    q_starts += [n.num_computed_tokens for so in ex.so_log for n in so.scheduled_new_reqs if n.req_id == "q"]
    assert ex.done_end["p"] > 0 and min(q_starts) > 0          # q's first chunk started after a cache hit


def test_planner_stale_done_is_ignored():
    planner = SlicePlanner(S0, S1, 256, min_tail=32)
    assert planner.complete(("x", 0)) is None
    planner.drop_request("x", "abort")
    assert planner.take_aborts() == ()


# ------------------------------------------------------------------------------------------- simulated clock
class _TimedFut(_Fut):
    def done(self) -> bool:
        return self.ex.end_at[self.idx] <= self.ex.now

    def result(self):
        self.ex.now = max(self.ex.now, self.ex.end_at[self.idx])
        return super().result()


# Measured on 8x V100-SXM2 (hol-design Appendix A, ms): decode verify step stage 0 32.6, stage 1 30 + post 6.8; prefill
# 1000 tokens stage 0 295 (14.8/layer), stage 1 228. Stage-1 slice split is NOT measured yet: two assumptions.
DEC0, DEC1 = 32.6e-3, 36.8e-3
PRE0, PRE1 = 0.295e-3, 0.228e-3            # per real prefill token on the normal path
LAYER0 = 14.8e-3
S1_SPLITS = {"s1 layer20 14.8 + rest 60 ms": (14.8e-3, 60e-3), "s1 layer20 30 + rest 198 ms": (30e-3, 198e-3)}
AL = 69.5 * (DEC0 + DEC1)                  # tokens per verify implied by the 69.5 tok/s steady decode


class TimedSlicedExecutor(SlicedPPExecutor):
    """Two FIFO stages. Each step: stage 0 runs decode (+ normal prefill), isends, runs its slices; stage 1 starts
    its decode when stage 0's decode part is done, then its slices (a ``first`` slice waits for stage 0's handoff).
    A step's output is ready when stage 1 ends (sampling waits for the slices: conservative)."""

    def __init__(self, scale, s1_cost, **kw):
        super().__init__(RANDOM, 0.8, kw.pop("seed", 0))
        self.scale, self.s1_cost = scale, s1_cost
        self.now = 0.0
        self.free = [0.0, 0.0]
        self.end_at: list[float] = []

    def _submit(self, fn):
        self.tasks.append(fn)
        self.results.append(None)
        self.end_at.append(self.end_at[-1] if self.end_at else 0.0)
        return _TimedFut(self, len(self.tasks) - 1)

    def execute_model(self, so, non_block=False):
        fut = super().execute_model(so, non_block)
        if not so.total_num_scheduled_tokens:
            self.end_at[fut.idx] = self.now
            return fut
        spec = so.scheduled_spec_decode_tokens
        decode = any(r in spec or (r.startswith("d") and n <= 6) for r, n in so.num_scheduled_tokens.items())
        pre = self.scale * sum(n for r, n in so.num_scheduled_tokens.items() if not r.startswith("d"))
        sl0 = sum(LAYER0 * (d.layer_end - d.layer_start) for d in so.ds41_slices if d.stage == 0)
        handoff = any(d.stage == 1 and d.first for d in so.ds41_slices)
        sl1 = sum(self._s1_cost(d) for d in so.ds41_slices if d.stage == 1)
        t0 = max(self.now, self.free[0])
        d0_end = t0 + (DEC0 if decode else 0.0) + PRE0 * pre
        self.free[0] = d0_end + sl0
        t1 = max(d0_end, self.free[1])
        d1_end = t1 + (DEC1 if decode else 0.0) + PRE1 * pre
        s1_start = max(d1_end, self.free[0]) if handoff else d1_end
        self.free[1] = s1_start + sl1
        self.end_at[fut.idx] = self.free[1]
        return fut

    def _s1_cost(self, d):
        """s1_cost = "model": the type-aware stage-1 cost at real scale (see _model_cost); (layer 20, rest) or (layer-20 attention, layer-20 MoE, rest of layers 21-39) for sub-layer
        slices; the rest is spread evenly over layers 21-39 (attention parts of those layers: their share / parts)."""
        if self.s1_cost == "model":
            return self._model_cost(d)
        if len(self.s1_cost) == 3:
            attn, moe, rest = self.s1_cost
            per = rest / (S1[1] - S1[0] - 1)
            if d.layer_part == 1:
                return (attn if d.layer_start == S1[0] else per) / d.part_count
            cost = 0.0
            for layer in range(d.layer_start, d.layer_end):
                if layer == S1[0]:
                    cost += moe if d.layer_part == 2 else attn + moe
                else:
                    cost += 0.0 if (layer == d.layer_start and d.layer_part == 2) else per
            return cost
        return self.s1_cost[0] if d.layer_start == S1[0] else self.s1_cost[1]

    # measured on 8x V100-SXM2 (hol-phase3 segment timing, T ~ 1016): layer-20 attention 12 ms + 1.5 ms per 1e6 (T x n_ctx),
    # plain layer 14 ms and reindex attention ~74 ms per 1024 rows the replay keeps (K = 3072), MoE ~60 % of a plain
    # layer, ~0.04 ms per skipped layer; + 2 ms per extra attention part (recomputed projections)
    REPLAY_K, PART_OVERHEAD = 3072, 2e-3

    def _model_cost(self, d):
        sc = self.scale
        T, start, end, n_req = d.num_tokens * sc, d.start_pos * sc, d.end_pos * sc, d.n_req * sc
        kept = max(0, end - max(start, n_req - self.REPLAY_K))
        reindex = {24, 28, 32, 36}

        def attn(layer):
            if layer == S1[0]:
                return 12e-3 + 1.5e-3 * T * end / 1e6
            if layer in reindex:
                return 74e-3 * kept / 1024
            return 0.4 * 14e-3 * kept / 1024

        def moe(layer):
            return 0.6 * 14e-3 * (T if layer == S1[0] else kept) / 1024

        if d.layer_part == 1:
            return attn(d.layer_start) / d.part_count + (self.PART_OVERHEAD if d.part_index else 0.0)
        cost = 0.0
        for layer in range(d.layer_start, d.layer_end):
            if layer == d.layer_start and d.layer_part == 2:
                cost += moe(layer)
            else:
                cost += attn(layer) + moe(layer) if (layer == S1[0] or kept) else 0.04e-3
        return cost

    def sample_tokens(self, grammar_output, non_block=False):
        fut = super().sample_tokens(grammar_output, non_block)
        self.end_at[fut.idx] = self.end_at[fut.idx - 1]
        return fut


def _timeline(monkeypatch, *, sliced, lane, s1_cost, s0_per=3, attn_parts=None):
    # 32K prompt = 2000 simulated tokens x 16; chunk 64 x 16 = 1024; normal-path tail 8 x 16 = 128
    scale = 16
    ex = TimedSlicedExecutor(scale, s1_cost)
    st = run(monkeypatch, sliced=sliced, lane=lane, executor=ex, long_prompt=2000, chunk=64, min_tail=8,
             dec_max_tokens=1500, max_model_len=2048, inject_after=4, s0_per=s0_per, attn_parts=attn_parts)
    info = st["info"]
    t0, t1 = info["t_inject"], info["t_first"]
    dec = [t for t, rid in info["arrivals"] if rid == "d0" and t0 < t <= t1]
    gaps = [b - a for a, b in zip([t0] + dec, dec + [t1])]
    return {"verify_hz": len(dec) / (t1 - t0), "prefill_tps": scale * 2000 / (t1 - t0), "max_gap": max(gaps),
            "dec_tps": AL * len(dec) / (t1 - t0)}


def test_timeline_32k_prefill(monkeypatch):
    rows = {}
    for name, s1_cost in S1_SPLITS.items():
        rows[f"sliced ({name})"] = _timeline(monkeypatch, sliced=True, lane=False, s1_cost=s1_cost)
    rows["sliced S0=4 layers (s1 14.8 + 60 ms)"] = _timeline(monkeypatch, sliced=True, lane=False,
                                                             s1_cost=next(iter(S1_SPLITS.values())), s0_per=4)
    rows["phase-0 lane (no slicing)"] = _timeline(monkeypatch, sliced=False, lane=True, s1_cost=(0, 0))
    rows["today (no lane)"] = _timeline(monkeypatch, sliced=False, lane=False, s1_cost=(0, 0))
    base = 1.0 / (DEC0 + DEC1)
    print(f"\n32K prefill with one B=1 decoder (AL {AL:.2f}/verify; idle decode = {base:.1f} verify/s = 69.5 tok/s)")
    for k, r in rows.items():
        print(f"  {k:36s} decoder {r['dec_tps']:5.1f} tok/s ({r['verify_hz']:5.2f} verify/s)  "
              f"prefill {r['prefill_tps']:6.0f} tok/s  max gap {1e3 * r['max_gap']:6.0f} ms")
    fast = rows[f"sliced ({next(iter(S1_SPLITS))})"]
    assert fast["dec_tps"] > 5 * rows["phase-0 lane (no slicing)"]["dec_tps"]
    assert fast["dec_tps"] >= 30 and fast["prefill_tps"] >= 1500


# ------------------------------------------------------------------------------------------- sub-layer slices
@pytest.mark.parametrize("seed", [0, 3])
@pytest.mark.parametrize("parts", [1, 3, "auto"])
@pytest.mark.parametrize("replay", [0, 600])
def test_s1_attention_parts_end_to_end(monkeypatch, seed, parts, replay):
    """VLLM_DS41_SLICE_S1_ATTN_PARTS: stage 1's heavy layers (layer 20, reindex layers with kept rows) run as
    attention parts then a MoE part; whole layers are packed by the type-aware cost model, never two reindex layers
    in one slice; the protocol fake checks the order, outputs stay exact, everything is freed."""
    monkeypatch.setenv("VLLM_DS41_INDEXER_QTILE_MAX", "32")      # 256-token chunks -> 8 scan tiles
    if replay:
        monkeypatch.setenv("VLLM_DS41_REPLAY_ROWS", str(replay))
    st = run(monkeypatch, seed=seed, attn_parts=parts)
    ex, planner = st["ex"], st["sched"]._ds41_slicer
    reindex = set(planner.reindex)
    by_layer: dict = {}
    for key, i, n in ex.part_log:
        by_layer.setdefault(key, []).append((i, n))
    assert ex.part_log or parts == "auto"
    s1 = [d for s in ex.slice_steps for d in s if d.stage == 1]
    for d in s1:
        if d.layer_part == 1:
            assert d.layer_start == S1[0] or d.layer_start in reindex, d
        else:
            whole = range(d.layer_start + (1 if d.layer_part == 2 else 0), d.layer_end)
            chunk = types.SimpleNamespace(start=d.start_pos, end=d.end_pos, num_tokens=d.num_tokens, n_req=d.n_req)
            heavy = [x for x in whole if x in reindex and planner._s1_rows(chunk, x) > 0]
            assert len(heavy) <= 1, (d, heavy)
    for s in ex.slice_steps:                     # one stage-1 slice per step, or chunk k's last + k+1's first
        mine = [d for d in s if d.stage == 1]
        assert len(mine) <= 1 or (len(mine) == 2 and mine[0].last and mine[1].first), mine


def test_s1_attention_parts_abort_mid_layer(monkeypatch):
    """A chunk aborted between its attention parts: both stages drop it (the fake drops the part cursor too)."""
    def mid_layer(st):
        return any(v != "moe" and v > 0 for v in st["ex"].parts[1].values() if isinstance(v, int))

    def abort_p(st):
        st["sched"].finish_requests("p", RequestStatus.FINISHED_ABORTED)
        st["aborted"].add("p")

    monkeypatch.setenv("VLLM_DS41_INDEXER_QTILE_MAX", "32")
    st = run(monkeypatch, attn_parts=3, events=[(mid_layer, abort_p)])
    assert st["ex"].aborts and not st["ex"].parts[1]


def test_attn_parts_auto_decision():
    """auto: parts from attn_fixed + scan_b * T * n_ctx over the budget, capped by the chunk's query-row tiles; a layer
    that fits one slice keeps today's whole-layer slice."""
    from vllm.v1.core.sched.ds41_sliced_prefill import _Chunk

    planner = SlicePlanner(S0, S1, 1024, min_tail=128, s1_attn_parts="auto")
    planner.s1_part_ms = 45.0
    short = _Chunk("r", 0, 0, 1024, (), 2048)           # 12 + 1.5 * 1024 * 1024 / 1e6 = 13.6 ms -> whole layer
    assert planner.attn_parts_for(short) == 0
    mid = _Chunk("r", 1, 31744, 1024, (), 40000)       # 12 + 1.5 * 1024 * 32768 / 1e6 = 62 ms -> 2 parts
    assert planner.attn_parts_for(mid) == 2
    long = _Chunk("r", 2, 130048, 1024, (), 200000)    # 12 + 1.5 * 1024 * 131072 / 1e6 = 213 ms -> 5 parts
    assert planner.attn_parts_for(long) == 5
    huge = _Chunk("r", 3, 1_000_000, 1024, (), 1_100_000)
    assert planner.attn_parts_for(huge) == 1024 // planner.qtile_cap       # capped by the tile count
    fixed = SlicePlanner(S0, S1, 1024, min_tail=128, s1_attn_parts=4)
    assert fixed.attn_parts_for(short) == 4


def test_policy_learns_scan_cost():
    p = AdaptiveSlicePolicy(keep=1.0, margin_ms=0.0, alpha=0.5)
    for _ in range(20):
        p.observe(slice_ms=[(1, -1, 12.0 + 2.0 * 33.5, False, 33.5), (1, 19, 38.0, False)])
    assert p.scan_b.value == pytest.approx(2.0) and p.per_layer[1].value == pytest.approx(2.0)


def test_timeline_32k_attention_parts(monkeypatch):
    """Simulated clock with the type-aware stage-1 cost at real scale (_model_cost: the hol-phase3 measurements;
    replay K = 3072, so the prompt's last ~3 chunks run layers 21-39 on every row): today's whole-layer stage-1 plan
    vs the sub-layer planner (auto). The planner sees the simulation's 16x-scaled tokens, so its priors are scaled
    the same way. Run with -s for the table."""
    sc = 16
    monkeypatch.setenv("VLLM_DS41_INDEXER_QTILE_MAX", str(128 // sc))
    rows = {"whole layers (today)": _timeline(monkeypatch, sliced=True, lane=False, s1_cost="model")}
    for k, v in (("VLLM_DS41_REPLAY_ROWS", 3072 // sc), ("VLLM_DS41_SLICE_S1_LAYER_MS", 14 * sc),
                 ("VLLM_DS41_SLICE_S1_REINDEX_ATTN_MS", 74 * sc), ("VLLM_DS41_SLICE_S1_SCAN_MS_PER_MTOK2", 1.5 * sc * sc)):
        monkeypatch.setenv(k, str(v))
    for budget in (45, 57):
        monkeypatch.setenv("VLLM_DS41_SLICE_S1_PART_MS", str(budget))
        rows[f"auto, budget {budget} ms"] = _timeline(monkeypatch, sliced=True, lane=False, s1_cost="model",
                                                     attn_parts="auto")
    print()
    for name, r in rows.items():
        print(f"{name:22s} decoder {r['dec_tps']:6.1f} tok/s  prefill {r['prefill_tps']:7.0f} tok/s  "
              f"max gap {r['max_gap'] * 1e3:6.1f} ms")
    assert rows["auto, budget 45 ms"]["max_gap"] <= rows["whole layers (today)"]["max_gap"] + 1e-9



def _walk(planner, chunk):
    kv = types.SimpleNamespace(get_block_ids_for_computed_tokens=lambda rid, n: ([1],))
    planner.s1 = chunk
    chunk.layer = S1[0]
    out = []
    while planner.s1 is not None:
        out.append(planner._advance(kv, chunk, 1))
    return out


@pytest.mark.parametrize("budget", [45.0, 60.0])
def test_s1_cost_model_kept_vs_skipped_chunks(monkeypatch, budget):
    """Real layer counts (20-39), T = 1024, decoder replay K = 3072: a chunk entirely past the cut (rows kept) runs
    layer 20 and each reindex layer as attention parts, every slice within the budget, <= 1 reindex layer per
    whole slice, every layer exactly once and in order; a chunk before the cut (replay skip) needs one slice."""
    from vllm.v1.core.sched.ds41_sliced_prefill import _Chunk

    monkeypatch.setenv("VLLM_DS41_REPLAY_ROWS", "3072")
    monkeypatch.setenv("VLLM_DS41_INDEXER_CANDIDATE_SCORING", "1")    # the live launcher's default: candidate scoring
    planner = SlicePlanner((0, 20), (20, 40), 1024, min_tail=128, s1_attn_parts="auto")
    planner.s1_part_ms = budget
    kept = _Chunk("r", 7, 30000, 1024, (), 32768)                 # positions 30000..31023 >= 32768 - 3072
    ds = _walk(planner, kept)
    seen = []
    for d in ds:
        if d.layer_part == 1:
            seen.append((d.layer_start, "attn", d.part_index, d.part_count))
            cost = planner.s1_attn_ms(kept, d.layer_start) / d.part_count
        else:
            a = d.layer_start
            cost = planner.s1_moe_ms(kept, a) if d.layer_part == 2 else planner.s1_layer_ms(kept, a)
            cost += sum(planner.s1_layer_ms(kept, x) for x in range(a + 1, d.layer_end))
            seen.append((a, d.layer_end, d.layer_part))
            assert sum(1 for x in range(a + (d.layer_part == 2), d.layer_end) if x in planner.reindex) <= 1
        multi = d.layer_part == 0 and d.layer_end - d.layer_start > 1
        assert cost <= budget + 1e-6 or not multi, (d, cost)
    attn_layers = sorted({x[0] for x in seen if x[1] == "attn"})
    assert attn_layers == [20, 24, 28, 32, 36]                     # 74 ms reindex attention > budget: parts
    for layer in (24, 28, 32, 36):
        parts = [x for x in seen if x[0] == layer and x[1] == "attn"]
        assert [p[2] for p in parts] == list(range(parts[0][3])) and parts[0][3] >= 2
    cover = []
    for d in ds:
        if d.layer_part != 1:
            cover += list(range(d.layer_start, d.layer_end))
    assert cover == list(range(20, 40)) and ds[0].first and ds[-1].last
    skipped = _Chunk("r", 0, 1024, 1024, (), 32768)                # far before the cut: replay skip past 20
    sd = _walk(planner, skipped)
    assert [(d.layer_start, d.layer_end, d.layer_part) for d in sd] == [(20, 40, 0)]


def test_policy_learns_per_layer_type_costs():
    p = AdaptiveSlicePolicy(keep=1.0, margin_ms=0.0, alpha=0.5)
    for _ in range(20):
        p.observe(slice_ms=[(1, -1, 92.5, False, 74.0, 2),           # reindex attention half: 92.5 ms vs prior 74
                            (1, 3, 45.0, False, 3.0, 3, 0),          # 3 plain layers x 1024 rows
                            (1, 2, 500.0, False, 1.0, 3, 1)])        # holds a reindex layer: not plain evidence
    assert p.reindex_ratio.value == pytest.approx(1.25) and p.plain.value == pytest.approx(15.0)


@pytest.mark.parametrize("parts", [3, "auto"])
def test_decoder_finishes_mid_attention_parts(monkeypatch, parts):
    """hol-phase3b's mixed_8k path: the decoder finishes while a replay-KEPT chunk is mid-way through stage 1's
    attention parts; the in-flight chunks are aborted and the prompt continues on the normal path. The protocol fake
    models per-stage KV through the block tables and checks every attended prefix, outputs are exact (run())."""
    monkeypatch.setenv("VLLM_DS41_INDEXER_QTILE_MAX", "32")
    monkeypatch.setenv("VLLM_DS41_REPLAY_ROWS", "600")
    aborted_mid = {}

    def check(st):
        ex = st["ex"]
        if any(isinstance(v, int) and v > 0 or v == "moe" for v in ex.parts[1].values()):
            aborted_mid["seen_parts_in_flight"] = True

    st = run(monkeypatch, dec_max_tokens=40, inject_after=8, long_prompt=1900, seed=4, attn_parts=parts, check=check)
    ex, slicer = st["ex"], st["sched"]._ds41_slicer
    assert ex.aborts and slicer.stats.by_reason.get("no_decoder", 0) > 0
    assert len([s["p"] for s in ex.normal_steps if "p" in s]) >= 2
    assert not ex.parts[1]                                       # the aborted chunk's part cursor is gone


def test_reindex_prior_follows_the_cand_prefill_mode(monkeypatch):
    """The planner's reindex attention prior follows VLLM_DS41_INDEXER_CAND_PREFILL: candidate scoring ~flat 74 ms per
    1024 rows; scan / auto below CAND_SCAN_MAX ~ 5.6 + 2.5 + 1.12 ms per 1K visible keys -> fewer attention parts."""
    from vllm.v1 import ds41_slices
    from vllm.v1.core.sched.ds41_sliced_prefill import _Chunk

    monkeypatch.setenv("VLLM_DS41_REPLAY_ROWS", "3072")
    monkeypatch.setenv("VLLM_DS41_INDEXER_CANDIDATE_SCORING", "1")
    kept = _Chunk("r", 7, 30000, 1024, (), 32768)
    monkeypatch.setenv("VLLM_DS41_INDEXER_CAND_PREFILL", "off")
    assert ds41_slices.reindex_attn_prior_ms(1024, 31024) == pytest.approx(74.0)
    off = SlicePlanner((0, 20), (20, 40), 1024, min_tail=128, s1_attn_parts="auto")
    assert off.attn_parts_for(kept, 24) == 2
    monkeypatch.setenv("VLLM_DS41_INDEXER_CAND_PREFILL", "auto")
    scan = ds41_slices.reindex_attn_prior_ms(1024, 31024)
    assert scan == pytest.approx(0.4 * 14 + 2.5 + 1.12 * 31.024)
    auto = SlicePlanner((0, 20), (20, 40), 1024, min_tail=128, s1_attn_parts="auto")
    auto.s1_part_ms = 60.0
    assert auto.attn_parts_for(kept, 24) == 0                    # ~43 ms + MoE fits one slice: whole layer
    far = _Chunk("r", 9, 60000, 1024, (), 62000)                 # n >= 49152: candidates again
    assert ds41_slices.reindex_attn_prior_ms(1024, far.end) == pytest.approx(74.0)
