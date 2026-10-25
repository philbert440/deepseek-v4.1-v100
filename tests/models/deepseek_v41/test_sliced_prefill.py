# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Layer-sliced prefill, model/worker half (VLLM_DS41_SLICED_PREFILL; vllm/v1/worker/ds41_sliced_prefill.py and
DeepseekV41Model.forward_slice). CPU only: a DeepseekV41Model shell whose layers are deterministic stand-ins that use
the stage's SharedAttnBuffers the way index sources / reuse layers do (write then read topk_indices) and write a fake
KV cache, so sliced execution with interleaved "decode" forwards can be compared bit for bit with an unsliced run.
The real kernels / NCCL handoff / builders are the GPU gate (scripts/hol_phase1_gpu.sh)."""

from __future__ import annotations

import contextlib
import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from vllm.model_executor.models.utils import PPMissingLayer
from vllm.models.deepseek_v41.common import contracts as C
from vllm.models.deepseek_v41.sm70.model import DeepseekV41Model
from vllm.v1.ds41_slices import SliceDirective
from vllm.v1.worker import ds41_sliced_prefill as sp

T_MAX = 48
N_LAYERS = 8          # stage 0 = [0, 4), stage 1 = [4, 8)
KV_ROWS = 4096


# ------------------------------------------------------------------------------------------------ stand-ins
class FakeLayer(nn.Module):
    """index-source layers (even ids) publish shared.topk_indices[:T]; every layer reads it; each layer writes one KV
    row per token into ``kv`` at ``slot_mapping`` of the forward context's metadata (when there is one)."""

    def __init__(self, idx: int, shared: C.SharedAttnBuffers, kv: torch.Tensor | None, md_name: str) -> None:
        super().__init__()
        self.idx, self.shared, self.kv, self.md_name = idx, shared, kv, md_name
        self.noise = 0.0              # != 0 makes the layer non-deterministic (verify must catch it)

    def forward(self, stream, pre, positions):
        T = stream.shape[0]
        if self.idx % 2 == 0:
            self.shared.topk_indices[:T, 0] = (stream[:, 0, :8].float().sum(-1) * 64).to(torch.int32) + self.idx
        k = self.shared.topk_indices[:T, 0].to(torch.float32)
        s = stream.float() * 0.75 + k[:, None, None] * 1e-3 + positions[:, None, None].float() * 1e-4 + self.idx
        md = _current_md()
        if md is not None and "swa" in md and md["swa"].window_slots is not None:
            s = s + md["swa"].window_slots[:T, :1, None].float() * 1e-5      # builder buffer read by every layer
        if self.noise:
            s = s + self.noise
            self.noise *= 2
        out = s.to(torch.bfloat16)
        if self.kv is not None:
            md = _current_md()
            if md is not None and self.md_name in md:
                slots = md[self.md_name].slot_mapping[:T]
                self.kv[self.idx].index_copy_(0, slots, out[:, 0, :16].float())
        return out, pre * 0.5 + 0.01 * self.idx


CTX = SimpleNamespace(attn_metadata=None)      # forward-context stand-in (model._forward_context() -> CTX)


def _current_md():
    return CTX.attn_metadata


@dataclasses.dataclass
class FakeMD:
    num_actual_tokens: int
    slot_mapping: torch.Tensor
    positions_cpu: np.ndarray
    window_slots: torch.Tensor | None = None


WIN_BUF = {0: torch.zeros(T_MAX, 4, dtype=torch.int64), 1: torch.zeros(T_MAX, 4, dtype=torch.int64)}   # per stage
R0 = 8                                 # VLLM_DS41_SLICE_SAVE_ROWS in these tests


class FakeEngram:
    def __init__(self) -> None:
        self.log: list = []
        self.lane = None
        self.next = 1

    def begin_slice(self, req_id, start_pos, num_tokens, prompt_token_ids=None, num_tokens_padded=None,
                    token_ids=None):
        assert token_ids is not None and len(token_ids) == num_tokens and num_tokens_padded == num_tokens
        h = self.next
        self.next += 1
        self.log.append(("begin", req_id, start_pos, num_tokens, h))
        return h

    @contextlib.contextmanager
    def slice_lane(self, h):
        assert self.lane is None
        self.lane = h
        self.log.append(("lane", h))
        try:
            yield
        finally:
            self.lane = None

    def end_slice(self, h):
        self.log.append(("end", h))


def make_model(first: bool, last: bool, start: int, end: int, kv: torch.Tensor | None = None,
               engram: FakeEngram | None = None) -> DeepseekV41Model:
    m = DeepseekV41Model.__new__(DeepseekV41Model)
    nn.Module.__init__(m)
    stage = C.StagePlan(pp_rank=0 if first else 1, pp_size=2, first_layer=start, last_layer=end - 1,
                        mirrored_kv_sources=(), exports_kv_sources=(),
                        engram_layers=(1,) if (first and engram is not None) else ())
    m.stage = stage
    m.pp_is_first, m.pp_is_last = first, last
    m.start_layer, m.end_layer = start, end
    m.shared = C.allocate_shared_attn_buffers(T_MAX, stage, torch.device("cpu"))
    m.layers = [FakeLayer(i, m.shared, kv, "swa") if start <= i < end else PPMissingLayer() for i in range(N_LAYERS)]
    m.replay = None
    m.mirror = None
    m.kv20_crc_check = False
    m._kv20_crc = None
    m.layer_subset = None
    m._engram_service = engram
    gen = torch.Generator().manual_seed(1)
    m.embed_tokens = nn.Embedding(1000, C.HIDDEN)
    with torch.no_grad():
        m.embed_tokens.weight.copy_(torch.randn(1000, C.HIDDEN, generator=gen))
    return m


def unsliced(m0, m1, ids: torch.Tensor, positions: torch.Tensor):
    """Reference: the whole chunk through both stages in one go each (fresh shared state)."""
    stream, pre = m0._embed_or_ingest(ids, positions, None)
    stream, pre, _, _, _ = m0._run_layers(stream, pre, positions, m0.start_layer, m0.end_layer)
    stream, pre, _, _, _ = m1._run_layers(stream, pre, positions, m1.start_layer, m1.end_layer)
    return stream, pre


def unsliced_with_md(m0, m1, ids, positions, d):
    """The unsliced reference under the chunk's (fresh, unclobbered) metadata on each stage."""
    stream, pre = m0._embed_or_ingest(ids, positions, None)
    CTX.attn_metadata = fake_md_builder(FakeRunner(), d, sp.DeviceArena(1 << 16, torch.device("cpu")), None)
    stream, pre, _, _, _ = m0._run_layers(stream, pre, positions, m0.start_layer, m0.end_layer)
    CTX.attn_metadata = fake_md_builder(FakeRunner(True), d, sp.DeviceArena(1 << 16, torch.device("cpu")), None)
    stream, pre, _, _, _ = m1._run_layers(stream, pre, positions, m1.start_layer, m1.end_layer)
    CTX.attn_metadata = None
    return stream, pre


def decode_clobber(m: DeepseekV41Model, runner=None, n: int = R0) -> None:
    """A decode/verify forward of ``n`` rows between slices: overwrites rows [0, n) of every per-step buffer -- the
    stage's shared buffers, the runner's PP receive buffers, the SWA builder's window_slots buffer."""
    m.shared.topk_indices[:n].fill_(777)
    m.shared.candidate_blocks[:n].fill_(-5)
    WIN_BUF[0 if m.pp_is_first else 1][:n].fill_(-9)
    it = getattr(runner, "intermediate_tensors", None)
    if it is not None:
        it[C.PP_KEY_HIDDEN][:n].fill_(3.0)
        it[C.PP_KEY_PRE_MIX][:n].fill_(-1.0)


# ------------------------------------------------------------------------------------------------ comm / runner
class Wire:
    """In-order point-to-point link stage 0 -> stage 1 (NCCL semantics: matched in issue order)."""

    def __init__(self) -> None:
        self.queue: list = []
        self.ops: list = []

    def sender(self):
        wire = self

        class _S:
            def isend_tensor_dict_static(self, d):
                wire.ops.append(("isend", tuple(d), tuple(tuple(t.shape) for t in d.values())))
                wire.queue.append({k: t.clone() for k, t in d.items()})
                return [SimpleNamespace(wait=lambda: wire.ops.append(("send_wait",)))]
        return _S()

    def receiver(self):
        wire = self

        class _R:
            def irecv_tensor_dict_static(self, views):
                wire.ops.append(("irecv", tuple(views), tuple(tuple(t.shape) for t in views.values())))

                def wait():
                    msg = wire.queue.pop(0)
                    assert tuple(msg) == tuple(views), (tuple(msg), tuple(views))
                    for k, v in views.items():
                        v.copy_(msg[k])
                    wire.ops.append(("recv_wait",))
                return [SimpleNamespace(wait=wait)]
        return _R()


class FakeRunner:
    def __init__(self, recv_buffers: bool = False) -> None:
        self.commits: list = []
        self.intermediate_tensors = None
        if recv_buffers:     # the persistent PP receive buffers of a non-first stage
            self.intermediate_tensors = {C.PP_KEY_HIDDEN: torch.zeros(T_MAX, C.HC, C.HIDDEN, dtype=torch.bfloat16),
                                         C.PP_KEY_PRE_MIX: torch.zeros(T_MAX, C.HC)}

    def _ds41_demand_kv_commit(self, extra, release):
        self.commits.append((extra, release))


def fake_md_builder(runner, d, arena, positions, lowrows=None):
    """Like build_chunk_metadata: slot mapping into the arena; window_slots stay in the (stage's) builder buffer."""
    T = d.num_tokens
    slots = sp.chunk_slot_mapping(d.block_ids[0], 16, d.start_pos, T)
    stage = 1 if runner.intermediate_tensors is not None else 0
    win = WIN_BUF[stage][:T]
    win.copy_(torch.arange(d.start_pos, d.start_pos + T)[:, None] * 3 + torch.arange(4))     # the builder's kernel
    md = FakeMD(T, torch.from_numpy(slots), np.arange(d.start_pos, d.start_pos + T), win)
    return {"swa": sp.clone_metadata(md, arena, {}, lowrows)}


@pytest.fixture(autouse=True)
def _md_context(monkeypatch):
    """forward_context stand-in: the executor's set_forward_context(md) -> CTX.attn_metadata; the model's
    _forward_context() -> CTX (what replay compaction swaps the metadata of)."""
    from vllm.models.deepseek_v41.sm70 import model as model_mod

    @contextlib.contextmanager
    def ctx(self, st):
        prev, CTX.attn_metadata = CTX.attn_metadata, st.md_active
        try:
            yield
        finally:
            CTX.attn_metadata = prev

    monkeypatch.setattr(sp.SlicedPrefillExecutor, "_forward_context", ctx)
    monkeypatch.setattr(model_mod, "_forward_context", lambda: CTX)
    monkeypatch.setattr(sp, "MIN_SLICE_TOKENS", 1)
    CTX.attn_metadata = None
    yield
    CTX.attn_metadata = None


def executors(kv=None, engram=None, slots1: int = 1, monkeypatch=None):
    if monkeypatch is not None:
        monkeypatch.setenv(sp.SLOTS_KNOB, str(slots1))
        monkeypatch.setenv(sp.ARENA_KNOB, "1")
        monkeypatch.setenv(sp.SAVE_ROWS_KNOB, str(R0))
    m0 = make_model(True, False, 0, 4, kv, engram)
    m1 = make_model(False, True, 4, 8, kv)
    wire = Wire()
    r0, r1 = FakeRunner(), FakeRunner(recv_buffers=True)
    ex0 = sp.SlicedPrefillExecutor(r0, m0, pp_rank=0, pp_size=2, max_tokens=T_MAX, device=torch.device("cpu"),
                                   comm=wire.sender(), build_metadata=fake_md_builder)
    ex1 = sp.SlicedPrefillExecutor(r1, m1, pp_rank=1, pp_size=2, max_tokens=T_MAX, device=torch.device("cpu"),
                                   comm=wire.receiver(), build_metadata=fake_md_builder)
    return m0, m1, ex0, ex1, wire


def directives(req: str, chunk: int, start: int, ids: np.ndarray, blocks, s0=((0, 2), (2, 4)), s1=((4, 5), (5, 8)),
               n_req: int = 500):
    T = len(ids)
    out0 = [SliceDirective(0, req, chunk, start, T, tuple(int(x) for x in ids), blocks, a, b, i == 0,
                           i == len(s0) - 1, n_req) for i, (a, b) in enumerate(s0)]
    out1 = [SliceDirective(1, req, chunk, start, T, (), blocks, a, b, i == 0, i == len(s1) - 1, n_req)
            for i, (a, b) in enumerate(s1)]
    return out0, out1


def step(ex0, ex1, m0, m1, slices=(), aborts=(), n_dec: int = R0):
    """One engine step on both stages, as the worker runs it: pre_step -> decode forward of n_dec rows (clobber) ->
    this stage's directives; stage 0 then stage 1."""
    so = SimpleNamespace(ds41_slices=tuple(slices), ds41_slice_aborts=tuple(aborts), total_num_scheduled_tokens=n_dec)
    ex0.pre_step(so)
    decode_clobber(m0, ex0.runner, n_dec)
    ex0.run_step(so)
    ex1.pre_step(so)
    decode_clobber(m1, ex1.runner, n_dec)
    return ex1.run_step(so)


# ------------------------------------------------------------------------------------------------ tests
def test_sliced_equals_unsliced_with_interleaved_decode(monkeypatch) -> None:
    m0, m1, ex0, ex1, wire = executors(monkeypatch=monkeypatch)
    T, start = 40, 32
    ids = np.arange(100, 100 + T, dtype=np.int32)
    blocks = (tuple(range(1, 6)),)
    d0, d1 = directives("A", 0, start, ids, blocks)
    done = []
    # stage 0 needs two steps; its 'last' and stage 1's 'first' ride the same step; stage 1 finishes a step later
    done += step(ex0, ex1, m0, m1, [d0[0]])
    done += step(ex0, ex1, m0, m1, [d0[1], d1[0]])
    assert not done
    done += step(ex0, ex1, m0, m1, [d1[1]])
    assert done == [("A", 0)]
    got_stream, got_pre = ex1.slots[0].stream[:T].clone(), ex1.slots[0].pre[:T].clone()
    pos = torch.arange(start, start + T)
    m0r = make_model(True, False, 0, 4)
    m1r = make_model(False, True, 4, 8)
    ref_stream, ref_pre = unsliced_with_md(m0r, m1r, torch.from_numpy(ids).long(), pos, d0[0])
    assert torch.equal(got_stream.view(torch.int16), ref_stream.view(torch.int16))
    assert torch.equal(got_pre, ref_pre)
    # the handoff went after stage 0's last slice, keys/shapes identical on both sides, in schema order
    sends = [o for o in wire.ops if o[0] == "isend"]
    recvs = [o for o in wire.ops if o[0] == "irecv"]
    assert len(sends) == len(recvs) == 1 and sends[0][1:] == recvs[0][1:]
    assert sends[0][1] == (C.PP_KEY_HIDDEN, C.PP_KEY_PRE_MIX, sp.HANDOFF_POSITIONS_KEY)
    assert sends[0][2] == ((T, C.HC, C.HIDDEN), (T, C.HC), (T,))
    # demand-KV commit of the chunk's blocks on each stage's 'first' only
    assert ex0.runner.commits == [([list(blocks[0])], None)] and ex1.runner.commits == [([list(blocks[0])], None)]
    # every slot free again, the model's shared buffers are its own (decode values), not the private ones
    assert not any(st.busy for st in ex0.slots + ex1.slots)
    assert m0.shared.topk_indices.data_ptr() != ex0.slots[0].shared.topk_indices.data_ptr()


def test_shared_buffers_restored_and_private_state_persists(monkeypatch) -> None:
    m0, m1, ex0, ex1, _ = executors(monkeypatch=monkeypatch)
    own = (m0.shared.topk_indices, m0.shared.candidate_blocks)
    ids = np.arange(5, 25, dtype=np.int32)
    d0, _ = directives("B", 3, 0, ids, ((1, 2),), s0=((0, 1), (1, 3), (3, 4)))
    ex0.run_step(SimpleNamespace(ds41_slices=(d0[0],)))
    assert (m0.shared.topk_indices, m0.shared.candidate_blocks) == own
    priv = ex0.slots[0].shared.topk_indices[:20, 0].clone()
    assert int(priv[0]) != 0                     # layer 0 (index source) published into the PRIVATE buffer
    decode_clobber(m0, None, T_MAX)
    assert torch.equal(ex0.slots[0].shared.topk_indices[:20, 0], priv)
    # stage 0 never touches candidate rows: no private copy; stage 1 owns no shared copy and no stream buffer at all
    assert ex0.slots[0].shared.candidate_blocks is m0.shared.candidate_blocks
    st1 = ex1.slots[0]
    assert st1.stream is None and st1.shared.topk_indices is m1.shared.topk_indices
    assert st1.shared.candidate_blocks is m1.shared.candidate_blocks


def test_stage1_receives_into_second_slot_while_busy(monkeypatch) -> None:
    m0, m1, ex0, ex1, wire = executors(slots1=2, monkeypatch=monkeypatch)
    ids_a, ids_b = np.arange(1, 21, dtype=np.int32), np.arange(31, 51, dtype=np.int32)
    a0, a1 = directives("A", 0, 0, ids_a, ((1, 2),), s0=((0, 4),))
    b0, b1 = directives("A", 1, 20, ids_b, ((1, 2, 3),), s0=((0, 4),))
    done = step(ex0, ex1, m0, m1, [a0[0], a1[0]])
    # stage 0 finishes chunk 1 while stage 1 still works on chunk 0: the recv lands in slot 2
    done += step(ex0, ex1, m0, m1, [b0[0], a1[1]])
    assert done == [("A", 0)]
    done += step(ex0, ex1, m0, m1, [b1[0]])
    done += step(ex0, ex1, m0, m1, [b1[1]])
    assert done == [("A", 0), ("A", 1)]


def test_single_slot_refuses_early_handoff(monkeypatch) -> None:
    m0, m1, ex0, ex1, _ = executors(slots1=1, monkeypatch=monkeypatch)
    a0, a1 = directives("A", 0, 0, np.arange(1, 21, dtype=np.int32), ((1, 2),), s0=((0, 4),))
    b0, _ = directives("A", 1, 20, np.arange(21, 41, dtype=np.int32), ((1, 2, 3),), s0=((0, 4),))
    step(ex0, ex1, m0, m1, [a0[0], a1[0]])
    with pytest.raises(RuntimeError, match="cannot receive the handoff"):
        step(ex0, ex1, m0, m1, [b0[0]])


def test_single_slot_handoff_waits_for_slot_within_the_step(monkeypatch) -> None:
    """Tuple order [stage-0 last k+1, stage-1 last k, stage-1 first k+1] on one stage-1 slot: the receive of k+1 is
    posted after stage 1 frees its slot, still in the same step (before the next step's decode payload)."""
    m0, m1, ex0, ex1, wire = executors(slots1=1, monkeypatch=monkeypatch)
    a0, a1 = directives("A", 0, 0, np.arange(1, 21, dtype=np.int32), ((1, 2),), s0=((0, 4),))
    b0, b1 = directives("A", 1, 20, np.arange(21, 41, dtype=np.int32), ((1, 2, 3),), s0=((0, 4),))
    done = step(ex0, ex1, m0, m1, [a0[0], a1[0]])
    done += step(ex0, ex1, m0, m1, [b0[0], a1[1], b1[0]])
    done += step(ex0, ex1, m0, m1, [b1[1]])
    assert done == [("A", 0), ("A", 1)]
    kinds = [o[0] for o in wire.ops if o[0] in ("isend", "irecv")]
    assert kinds == ["isend", "irecv", "isend", "irecv"]


def test_stage1_first_without_handoff_and_layer_gaps_raise(monkeypatch) -> None:
    m0, m1, ex0, ex1, _ = executors(monkeypatch=monkeypatch)
    d0, d1 = directives("A", 0, 0, np.arange(1, 21, dtype=np.int32), ((1, 2),))
    with pytest.raises(RuntimeError, match="without its handoff"):
        ex1.run_step(SimpleNamespace(ds41_slices=(d1[0],)))
    ex0.run_step(SimpleNamespace(ds41_slices=(d0[0],)))
    gap = dataclasses.replace(d0[1], layer_start=3)
    with pytest.raises(RuntimeError, match="chunk is at layer 2"):
        ex0.run_step(SimpleNamespace(ds41_slices=(gap,)))


def test_abort_paths(monkeypatch) -> None:
    eng = FakeEngram()
    m0, m1, ex0, ex1, wire = executors(engram=eng, monkeypatch=monkeypatch)
    ids = np.arange(1, 21, dtype=np.int32)
    d0, d1 = directives("A", 0, 0, ids, ((1, 2),), s0=((0, 1), (1, 4)))
    # (1) abort on stage 0 mid-chunk, before the Engram layer: the Engram slice ends, the slot frees
    ex0.run_step(SimpleNamespace(ds41_slices=(d0[0],)))
    h = eng.log[0][-1]
    ex0.run_step(SimpleNamespace(ds41_slices=(), ds41_slice_aborts=(("A", 0),)))
    assert ("end", h) in eng.log and not ex0.slots[0].busy and ex0.stats["aborts"] == 1
    # unknown aborts are ignored
    ex0.run_step(SimpleNamespace(ds41_slices=(), ds41_slice_aborts=(("Z", 9),)))
    # (2) abort on stage 1 after the handoff was posted but before its first slice: the recv is drained
    step(ex0, ex1, m0, m1, [d0[0]])
    step(ex0, ex1, m0, m1, [d0[1]])
    assert ex1.slots[0].busy and ex1.slots[0].recv_handles
    step(ex0, ex1, m0, m1, [], aborts=[("A", 0)])
    assert not ex1.slots[0].busy and not wire.queue and ("recv_wait",) in wire.ops
    # (3) the Engram slice ended once, right after the slice that ran the last Engram layer (layer 1)
    ends = [e for e in eng.log if e[0] == "end"]
    assert len(ends) == 2
    lanes = [e for e in eng.log if e[0] == "lane"]
    assert len(lanes) == 3      # chunk 1: [0,1) ; chunk 2: [0,1), [1,4)


def test_verify_detects_mismatch_and_passes_when_deterministic(monkeypatch) -> None:
    monkeypatch.setenv(sp.VERIFY_KNOB, "1")
    kv = torch.zeros(N_LAYERS, KV_ROWS, 16)
    m0, m1, ex0, ex1, _ = executors(kv=kv, monkeypatch=monkeypatch)
    for ex in (ex0, ex1):
        views = lambda st, ids, ex=ex: [(f"model.layers.{i}.attn.swa", kv[i].view(-1, 16, 16),
                                         torch.tensor(list(ids[0])[st.start_pos // 16:(st.start_pos + st.num_tokens - 1) // 16 + 1]))
                                        for i in range(ex.stage_first, ex.stage_end)]
        monkeypatch.setattr(ex, "_chunk_kv_views", views)
    errors: list = []
    monkeypatch.setattr(sp.logger, "error", lambda msg, *a: errors.append(msg % a))
    ids = np.arange(3, 43, dtype=np.int32)
    d0, d1 = directives("V", 0, 16, ids, ((1, 2, 3, 4),))
    done = step(ex0, ex1, m0, m1, [d0[0]]) + step(ex0, ex1, m0, m1, [d0[1], d1[0]]) + step(ex0, ex1, m0, m1, [d1[1]])
    assert done == [("V", 0)] and errors == [] and ex0.stats["verified"] == ex1.stats["verified"] == 1
    m1.layers[6].noise = 1e-1               # non-deterministic layer on stage 1
    d0, d1 = directives("V", 1, 56, np.arange(50, 90, dtype=np.int32), ((1, 2, 3, 4, 5, 6, 7),))
    step(ex0, ex1, m0, m1, [d0[0]]), step(ex0, ex1, m0, m1, [d0[1], d1[0]]), step(ex0, ex1, m0, m1, [d1[1]])
    assert ex1.stats["verify_fail"] == 1 and ex0.stats["verify_fail"] == 0
    assert errors and all("slice verify MISMATCH" in e for e in errors)
    import re
    assert any(re.search(r"slice.?verify.*(mismatch|differ|FAIL)", e, re.I) for e in errors)
    assert any("layer=6" in e for e in errors) and any("tensor=stream" in e for e in errors)


def test_attach_slice_done_never_mutates_empty_output() -> None:
    from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT

    out = sp.attach_slice_done(EMPTY_MODEL_RUNNER_OUTPUT, [("A", 0)])
    assert out is not EMPTY_MODEL_RUNNER_OUTPUT and tuple(out.ds41_slice_done) == (("A", 0),)
    assert not tuple(getattr(EMPTY_MODEL_RUNNER_OUTPUT, "ds41_slice_done", ()) or ())
    out2 = sp.attach_slice_done(out, [("B", 2)])
    assert tuple(out2.ds41_slice_done) == (("A", 0), ("B", 2))
    assert sp.attach_slice_done(None, [("A", 0)]) is None
    import pickle
    assert tuple(pickle.loads(pickle.dumps(out2)).ds41_slice_done) == (("A", 0), ("B", 2))


def test_chunk_slot_mapping_and_arena() -> None:
    slots = sp.chunk_slot_mapping([7, 3, 9], 16, 20, 20)
    pos = np.arange(20, 40)
    assert np.array_equal(slots, np.array([7, 3, 9])[pos // 16] * 16 + pos % 16)
    with pytest.raises(ValueError):
        sp.chunk_slot_mapping([7], 16, 0, 20)
    arena = sp.DeviceArena(1024, torch.device("cpu"))
    a = arena.take(torch.arange(10, dtype=torch.int64))
    assert arena.owns(a) and a.data_ptr() % 1 == 0 and torch.equal(a, torch.arange(10))
    with pytest.raises(RuntimeError, match="raise VLLM_DS41_SLICE_MD_ARENA_MIB"):
        arena.take(torch.zeros(1000, dtype=torch.int64))             # overflow: clean error, never a silent clone


def test_clone_metadata_detaches_from_builder_buffers() -> None:
    builder_buf = torch.arange(64, dtype=torch.int64)
    md = FakeMD(8, builder_buf[:8], np.arange(8))
    arena = sp.DeviceArena(4096, torch.device("cpu"))
    memo: dict = {}
    c = sp.clone_metadata(md, arena, memo)
    assert sp.clone_metadata(md, arena, memo) is c          # shared metadata objects are cloned once
    builder_buf.fill_(-1)                                   # the next decode step rebuilds into the builder buffer
    md.positions_cpu[:] = -1
    assert torch.equal(c.slot_mapping, torch.arange(8)) and np.array_equal(c.positions_cpu, np.arange(8))
    assert arena.owns(c.slot_mapping)


def test_build_chunk_metadata_uses_runner_builders(monkeypatch) -> None:
    """One request [start, start+T): qsl [0,T], seq [start+T], block table = the directive's ids, slot mapping per
    group with that group's block size; layer names map to arena-owned clones; non-DS41 builders are skipped."""
    seen: list = []

    class Builder:
        def __init__(self, keep=True):
            self.buf = torch.zeros(T_MAX, dtype=torch.int64)
            self.keep = keep

        def build(self, common_prefix_len, common_attn_metadata):
            cm = common_attn_metadata
            seen.append(cm)
            T = cm.num_actual_tokens
            self.buf[:T] = cm.slot_mapping[:T] + 1
            return FakeMD(T, self.buf[:T], np.arange(T))

    b0, b1, skip = Builder(), Builder(), Builder(keep=False)
    group = lambda bs: SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=bs))
    runner = SimpleNamespace(
        kv_cache_config=SimpleNamespace(kv_cache_groups=[group(16), group(32)]),
        attn_groups=[[SimpleNamespace(get_metadata_builder=lambda i: b0, layer_names=["l0.swa", "l1.swa"])],
                     [SimpleNamespace(get_metadata_builder=lambda i: b1, layer_names=["l2.ckv"]),
                      SimpleNamespace(get_metadata_builder=lambda i: skip, layer_names=["draft"])]])
    monkeypatch.setattr(sp, "_builders_eager", contextlib.nullcontext)
    d = SliceDirective(0, "A", 0, 40, 24, tuple(range(24)), ((5, 6, 7, 8), (9, 10)), 0, 2, True, False, 100)
    arena = sp.DeviceArena(1 << 16, torch.device("cpu"))
    md = sp.build_chunk_metadata(runner, d, arena, torch.arange(40, 64), is_builder=lambda b: b.keep)
    assert set(md) == {"l0.swa", "l1.swa", "l2.ckv"} and md["l0.swa"] is md["l1.swa"]
    cm0, cm1 = seen
    assert cm0.num_reqs == 1 and cm0.num_actual_tokens == 24 and cm0.max_seq_len == 64
    assert cm0.query_start_loc_cpu.tolist() == [0, 24] and cm0.seq_lens.tolist() == [64]
    assert cm0.block_table_tensor.tolist() == [[5, 6, 7, 8]] and cm1.block_table_tensor.tolist() == [[9, 10]]
    pos = np.arange(40, 64)
    assert cm0.slot_mapping.tolist() == (np.array([5, 6, 7, 8])[pos // 16] * 16 + pos % 16).tolist()
    assert cm1.slot_mapping.tolist() == (np.array([9, 10])[pos // 32] * 32 + pos % 32).tolist()
    b0.buf.fill_(0)
    assert md["l0.swa"].slot_mapping.tolist() == (cm0.slot_mapping + 1).tolist() and arena.owns(md["l0.swa"].slot_mapping)


def test_engram_planner_resolves_tail_after_staged_chunks() -> None:
    """The normal-path tail of a request whose first chunks were staged carries token ids (EngramReqStep)."""
    from vllm.v1.worker.gpu_worker import _EngramStepPlanner

    prompt = np.arange(1000, 1100, dtype=np.int32)
    pl = _EngramStepPlanner()
    pl.note_staged("A", 0, prompt[:40])
    pl.note_staged("A", 40, prompt[40:80])
    cached = SimpleNamespace(req_ids=["A"], num_computed_tokens=[80], resumed_req_ids=set(), all_token_ids={},
                             new_token_ids=[])
    so = SimpleNamespace(scheduled_new_reqs=[], scheduled_cached_reqs=cached, num_scheduled_tokens={"A": 20},
                         scheduled_spec_decode_tokens={}, finished_req_ids=set(), preempted_req_ids=set())
    # without new_token_ids (PP=1) the planner only knows the staged ids: the tail beyond them is unresolved
    with pytest.raises(NotImplementedError):
        pl.plan(so)
    # under PP (the deployed mode) the scheduler sends the prompt continuation as new_token_ids
    cached.new_token_ids = [prompt[80:100].tolist()]
    plan = pl.plan(so)
    assert np.array_equal(plan.reqs[0].token_ids, prompt[80:100]) and plan.reqs[0].start_pos == 80
    # staged chunks extend, never truncate, a longer known prompt
    pl2 = _EngramStepPlanner()
    pl2._tokens["B"] = prompt.copy()
    pl2.note_staged("B", 0, prompt[:40])
    assert np.array_equal(pl2._tokens["B"], prompt)


def test_forward_slice_rejects_bad_ranges(monkeypatch) -> None:
    m0 = make_model(True, False, 0, 4)
    st = SimpleNamespace(shared=m0.shared, engram_handle=None, engram_service=None)
    with pytest.raises(ValueError, match="outside this stage"):
        m0.forward_slice(st, 2, 6)


class StubReplay:
    """Decoder-replay stand-in: plan = keep rows with position >= n_req - k (host only), compaction by index_select."""

    def __init__(self, k: int) -> None:
        self.k, self.first_layer = k, 5
        self.binding = None
        self.zeroed: list = []

    def bind(self, num_reqs, num_tokens, qsl, n_req):
        self.binding = (num_reqs, num_tokens, n_req)

    def begin(self, md):
        _, T, n_req = self.binding
        self.binding = None
        pos = md["swa"].positions_cpu
        keep = np.flatnonzero(pos >= int(n_req[0]) - self.k)
        if keep.shape[0] == T:
            return None
        return SimpleNamespace(kind="skip" if keep.shape[0] == 0 else "compact", kept=keep)

    def zero_dropped(self, md, dropped):
        self.zeroed.append(None if dropped is None else dropped.tolist())

    def note_skip(self, plan):
        pass

    def compact_step(self, md, plan, shared):
        kept = torch.from_numpy(plan.kept)
        shared.topk_indices[:kept.shape[0]].copy_(shared.topk_indices.index_select(0, kept))
        md2 = {"swa": FakeMD(int(kept.shape[0]), md["swa"].slot_mapping.index_select(0, kept),
                             md["swa"].positions_cpu[plan.kept])}
        dropped = torch.from_numpy(np.setdiff1d(np.arange(md["swa"].num_actual_tokens), plan.kept))
        return md2, SimpleNamespace(kept=kept, dropped=dropped, positions=torch.from_numpy(md2["swa"].positions_cpu))


@pytest.mark.parametrize("n_req,kind", [(60, "compact"), (5000, "skip"), (40, "none")])
def test_forward_slice_replay_matches_unsliced(monkeypatch, n_req: int, kind: str) -> None:
    """Stage 1 with decoder replay: slice [4,5) then [5,8) (compaction at layer 5 persisted in the state) equals the
    unsliced forward's plan on the same chunk."""
    m0, m1, ex0, ex1, _ = executors(monkeypatch=monkeypatch)
    m1.replay = StubReplay(k=30)
    ids = np.arange(1, 25, dtype=np.int32)
    d0, d1 = directives("R", 0, 10, ids, ((1, 2, 3),), s0=((0, 4),), s1=((4, 5), (5, 6), (6, 8)), n_req=n_req)
    step(ex0, ex1, m0, m1, [d0[0], d1[0]])
    st = ex1.slots[0]
    # the handoff input, re-run unsliced with the same plan
    inp_stream, inp_pre = ex0.slots[0].stream[:24].clone(), ex0.slots[0].pre[:24].clone()
    step(ex0, ex1, m0, m1, [d1[1]])
    assert st.replay_done and st.replay_skip == (kind == "skip")
    if kind == "compact":
        assert st.rows < 24 and st.md_active is not st.md
    got_rows = st.rows                      # the slot is cleared at 'last' (buffers keep the values)
    done = step(ex0, ex1, m0, m1, [d1[2]])
    assert done == [("R", 0)]
    got = st.stream[:got_rows].clone()

    ref = make_model(False, True, 4, 8)
    ref.replay = StubReplay(k=30)
    pos = torch.arange(10, 34)
    md = fake_md_builder(FakeRunner(True), d1[0], sp.DeviceArena(1 << 16, torch.device("cpu")), None)
    CTX.attn_metadata = md
    ref.replay.bind(1, 24, np.array([0, 24]), np.array([n_req]))
    plan = ref.replay.begin(md)
    if plan is not None:
        plan = SimpleNamespace(kind=plan.kind, kept=plan.kept)
    s, p, _, _, _ = ref._run_layers(inp_stream, inp_pre, pos, 4, 8, plan=plan)
    assert CTX.attn_metadata is md          # the unsliced loop restores the full metadata
    if kind == "skip":
        s = s[:24]                      # layers 5+ never ran: the stream after layer 4
    assert got_rows == s.shape[0] and torch.equal(got.view(torch.int16), s.view(torch.int16))


def test_stage1_lives_in_pp_recv_buffers_and_survives_decode_steps(monkeypatch) -> None:
    """Stage 1 holds the chunk across steps in the runner's PP receive buffers + the model's shared buffers; decode
    steps (n_dec <= R0) clobber the low rows, pre_step/run_step save and restore them; a main batch > R0 evacuates."""
    m0, m1, ex0, ex1, _ = executors(monkeypatch=monkeypatch)
    T, start = 40, 32
    ids = np.arange(100, 100 + T, dtype=np.int32)
    d0, d1 = directives("A", 0, start, ids, (tuple(range(1, 6)),), s1=((4, 5), (5, 6), (6, 8)))
    done = step(ex0, ex1, m0, m1, [d0[0]])
    done += step(ex0, ex1, m0, m1, [d0[1], d1[0]])
    st = ex1.slots[0]
    assert st.stream.data_ptr() == ex1.runner.intermediate_tensors[C.PP_KEY_HIDDEN].data_ptr()
    done += step(ex0, ex1, m0, m1, [d1[1]], n_dec=R0)          # decode-sized step: low rows saved/restored
    done += step(ex0, ex1, m0, m1, [d1[2]], n_dec=T)           # prefill-sized main batch: evacuation
    assert done == [("A", 0)] and ex1._evac_warned == 1
    got = st.stream[:T].clone() if st.stream is not None else ex1.runner.intermediate_tensors[C.PP_KEY_HIDDEN][:T].clone()
    pos = torch.arange(start, start + T)
    ref_stream, _ = unsliced_with_md(make_model(True, False, 0, 4), make_model(False, True, 4, 8),
                                     torch.from_numpy(ids).long(), pos, d0[0])
    assert torch.equal(got.view(torch.int16), ref_stream.view(torch.int16))


def test_save_rows_must_cover_decode_capture_sizes(monkeypatch) -> None:
    monkeypatch.setenv(sp.SAVE_ROWS_KNOB, "16")
    cfg = SimpleNamespace(compilation_config=SimpleNamespace(cudagraph_capture_sizes=[1, 2, 4, 8, 16, 24]))
    m0 = make_model(True, False, 0, 4)
    with pytest.raises(ValueError, match="largest decode capture size 24"):
        sp.SlicedPrefillExecutor(FakeRunner(), m0, pp_rank=0, pp_size=2, max_tokens=T_MAX, device=torch.device("cpu"),
                                 vllm_config=cfg)


def test_low_row_save_buffer_overflow_is_a_clean_error(monkeypatch) -> None:
    monkeypatch.setenv(sp.SAVE_KNOB, "0.01")                   # ~10 KiB: too small for 8 rows of the stream
    m0, m1, ex0, ex1, _ = executors(monkeypatch=monkeypatch)
    d0, d1 = directives("A", 0, 0, np.arange(1, 21, dtype=np.int32), ((1, 2),), s0=((0, 4),))
    with pytest.raises(RuntimeError, match="raise VLLM_DS41_SLICE_SAVE_MIB"):
        step(ex0, ex1, m0, m1, [d0[0], d1[0]])


class _FakeSlicer:
    def __init__(self, is_last: bool) -> None:
        self.is_first, self.is_last, self.pp_size = not is_last, is_last, 2
        self.on_staged = None
        self.pending_done: list = []
        self.calls: list = []

    def run_step(self, so):
        self.calls.append(so)
        return [(d.req_id, d.chunk_id) for d in so.ds41_slices if d.last and d.stage == 1]

    def pre_step(self, so):
        pass


def _worker(slicer, monkeypatch, after_sample="1", no_empty="0"):
    from vllm.v1.outputs import ModelRunnerOutput
    from vllm.v1.worker.gpu_worker import Worker

    monkeypatch.setenv("VLLM_DS41_SLICE_S1_AFTER_SAMPLE", after_sample)
    monkeypatch.setenv("VLLM_DS41_PP_NO_EMPTY_STEP", no_empty)
    w = Worker.__new__(Worker)
    w.model_runner = SimpleNamespace(_ds41_slicer=slicer,
                                     sample_tokens=lambda g: ModelRunnerOutput(req_ids=["d"], req_id_to_index={"d": 0}))
    return w


def test_last_stage_slices_run_after_the_sampled_output(monkeypatch) -> None:
    """Stage 1: execute_model returns None (spec decode) -> the step's slices are deferred; they do not run after the
    execute_model response, they run after sample_tokens' response; their slice_done rides the next sampled output.
    Never mutates EMPTY_MODEL_RUNNER_OUTPUT."""
    from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT

    sl = _FakeSlicer(is_last=True)
    w = _worker(sl, monkeypatch)
    d = SliceDirective(1, "A", 0, 0, 32, (), ((1,),), 5, 8, False, True, 100)
    so = SimpleNamespace(ds41_slices=(d,), ds41_slice_aborts=())
    assert w._ds41_run_slices(so, None) is None and not sl.calls
    w.ds41_after_response()                               # after execute_model's (None) response: not yet
    assert not sl.calls
    out1 = w.sample_tokens(None)
    assert not tuple(getattr(out1, "ds41_slice_done", ()))
    w.ds41_after_response()                               # after sample_tokens' response: the slices run
    assert sl.calls == [so] and sl.pending_done == [("A", 0)]
    out2 = w.sample_tokens(None)                          # next step's sampled output carries the done
    assert tuple(out2.ds41_slice_done) == (("A", 0),) and not sl.pending_done
    # an empty step (aborts only) whose output is EMPTY: deferred too, then flushed; the constant stays untouched
    so2 = SimpleNamespace(ds41_slices=(), ds41_slice_aborts=(("A", 1),))
    assert w._ds41_run_slices(so2, EMPTY_MODEL_RUNNER_OUTPUT) is EMPTY_MODEL_RUNNER_OUTPUT
    w.ds41_after_response()
    assert sl.calls[-1] is so2 and not tuple(getattr(EMPTY_MODEL_RUNNER_OUTPUT, "ds41_slice_done", ()) or ())
    # flush at the next execute_model if the busy loop hook never ran
    w._ds41_run_slices(so, EMPTY_MODEL_RUNNER_OUTPUT)
    w._ds41_flush_deferred()
    assert sl.calls[-1] is so


def test_after_sample_off_with_no_empty_step(monkeypatch) -> None:
    sl = _FakeSlicer(is_last=True)
    w = _worker(sl, monkeypatch, no_empty="1")
    d = SliceDirective(1, "A", 0, 0, 32, (), ((1,),), 5, 8, False, True, 100)
    w._ds41_run_slices(SimpleNamespace(ds41_slices=(d,), ds41_slice_aborts=()), None)
    assert len(sl.calls) == 1 and sl.pending_done == [("A", 0)]     # inline, done on this step's sampled output
    assert tuple(w.sample_tokens(None).ds41_slice_done) == (("A", 0),)


# ------------------------------------------------------------------------------------------------ adaptive feedback
def test_feedback_pack_unpack_and_clean_flag() -> None:
    from vllm.v1.worker.ds41_slice_feedback import MAX_SLICES, SliceFeedback

    f0 = SliceFeedback(0)
    for i in range(MAX_SLICES + 3):
        f0.note_slice(3, 40.0 + i, i == 0)
    f0.note_decode("decode", False, 32.0, 0.0, 32.0)
    f0.note_decode("decode", True, 34.0, 0.0, 34.0)
    f0.note_decode("prefill", False, 300.0, 0.0, 300.0)            # not a decode step: ignored
    v = f0.pack()
    assert not f0.slices and f0.pack()[-3] == 0.0                     # drained
    f1 = SliceFeedback(1)
    f1.unpack([0.0] * len(v))                                         # a non-TP0 rank's zeros: nothing
    f1.unpack(v)
    s, d = f1.take()
    assert len(s) == MAX_SLICES and s[-1] == (0, 3, 40.0 + MAX_SLICES + 2, False)      # newest kept
    assert d == ((0, 33.0, 0.0, False),)
    # last stage: fwd + post, clean only when neither this nor the previous step carried slices
    f1.note_decode("decode", False, 30.0, 7.0, 70.0)
    f1.note_decode("decode", True, 30.0, 7.0, 90.0)
    f1.note_decode("decode", False, 30.0, 7.0, 80.0)
    f1.note_decode("decode", False, 30.0, 7.0, 71.0)
    assert [x[3] for x in f1.take()[1]] == [True, False, False, True]


def test_adaptive_feedback_rides_the_handoff(monkeypatch) -> None:
    """VLLM_DS41_SLICE_POLICY=adaptive: the handoff carries one extra f32 tensor (last key, both sides); the last stage
    reports stage 0's slice samples and its own; the chunk itself is unchanged (bit-equal to the knob-off run)."""
    from vllm.v1 import ds41_slices

    def run_chunk(adaptive: bool):
        if adaptive:
            monkeypatch.setenv(ds41_slices.SLICED_PREFILL_KNOB, "1")
            monkeypatch.setenv(ds41_slices.SLICE_POLICY_KNOB, "adaptive")
        else:
            monkeypatch.delenv(ds41_slices.SLICE_POLICY_KNOB, raising=False)
        m0, m1, ex0, ex1, wire = executors(monkeypatch=monkeypatch)
        T, start = 40, 32
        ids = np.arange(100, 100 + T, dtype=np.int32)
        d0, d1 = directives("A", 0, start, ids, (tuple(range(1, 6)),))
        step(ex0, ex1, m0, m1, [d0[0]])
        step(ex0, ex1, m0, m1, [d0[1], d1[0]])
        done = step(ex0, ex1, m0, m1, [d1[1]])
        assert done == [("A", 0)]
        return ex0, ex1, wire, ex1.slots[0].stream[:T].clone()

    ex0, ex1, wire, stream_on = run_chunk(True)
    send = [o for o in wire.ops if o[0] == "isend"][0]
    recv = [o for o in wire.ops if o[0] == "irecv"][0]
    assert send[1:] == recv[1:] and send[1][-1] == ds41_slices.SLICE_FB_KEY
    assert send[2][-1] == (ds41_slices.SLICE_FB_LEN,)
    slice_ms, decode_ms = ex1.take_feedback()
    assert [(s, n, f) for s, n, _, f in slice_ms] == [(0, 2, True), (0, 2, False), (1, 1, True), (1, 3, False)]
    assert all(ms >= 0 for _, _, ms, _ in slice_ms) and decode_ms == ()
    assert ex0.take_feedback() == ((), ())                     # stage 0 shipped its samples
    _, _, wire_off, stream_off = run_chunk(False)
    assert [o for o in wire_off.ops if o[0] == "isend"][0][1][-1] == sp.HANDOFF_POSITIONS_KEY
    assert torch.equal(stream_on.view(torch.int16), stream_off.view(torch.int16))


def test_attach_feedback_never_mutates_empty_output() -> None:
    from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT

    out = sp.attach_feedback(EMPTY_MODEL_RUNNER_OUTPUT, [(1, 2, 5.0, False)], [(1, 37.0, 70.0, True)])
    assert out is not EMPTY_MODEL_RUNNER_OUTPUT and out.ds41_slice_ms == ((1, 2, 5.0, False),)
    assert out.ds41_decode_ms == ((1, 37.0, 70.0, True),) and EMPTY_MODEL_RUNNER_OUTPUT.ds41_slice_ms == ()
    assert sp.attach_feedback(EMPTY_MODEL_RUNNER_OUTPUT, (), ()) is EMPTY_MODEL_RUNNER_OUTPUT


# ------------------------------------------------------------------------------------------------ sub-layer slices
TILE = 8      # the fake scan's query-row tile


class FakeSplitLayer(FakeLayer):
    """Stand-in with the decoder layer's attn_half / moe_half split and an attention whose "scan" publishes
    shared.topk_indices[:, 1] tile by tile (each attention part runs its share of the tiles, like the indexer's
    scan_tasks); part 0 alone writes the layer's KV rows; the last part reads every row's scan result."""

    def _mix(self, stream, pre_in):
        return (stream.float() * 0.5 + pre_in.sum(-1)[:, None, None] * 0.1).to(torch.bfloat16)

    def attn_half(self, stream, pre_in, positions, part=None):
        T = stream.shape[0]
        i, n = part if part is not None else (0, 1)
        x = self._mix(stream, pre_in)                               # recomputed every part from the same inputs
        tiles = [(a, min(T, a + TILE)) for a in range(0, T, TILE)]
        from vllm.v1.ds41_slices import split_range
        lo, hi = split_range(len(tiles), i, n)
        for a, b in tiles[lo:hi]:            # an index source: publishes the top-k the next layers read (col 0 too)
            v = (x[a:b, 0, :4].float().sum(-1) * 32).to(torch.int32) + self.idx
            self.shared.topk_indices[a:b, 1] = v
            self.shared.topk_indices[a:b, 0] = v + 1
        if i == 0 and self.kv is not None:
            md = _current_md()
            if md is not None and "swa" in md:
                self.kv[self.idx].index_copy_(0, md["swa"].slot_mapping[:T], x[:, 0, :16].float())
        if i < n - 1:
            return None
        k = self.shared.topk_indices[:T, 1].float()
        if self.noise:
            k = k + self.noise
            self.noise *= 2
        out = (x.float() + k[:, None, None] * 1e-3 + positions[:, None, None].float() * 1e-4).to(torch.bfloat16)
        return out, pre_in * 0.25 + 0.5

    def moe_half(self, stream, attn_pre):
        return (stream.float() * 0.9 + attn_pre.mean(-1)[:, None, None]).to(torch.bfloat16), attn_pre * 0.5 - 0.1

    def forward(self, stream, pre, positions):
        return self.moe_half(*self.attn_half(stream, pre, positions))


def split_executors(monkeypatch, kv=None):
    m0, m1, ex0, ex1, wire = executors(kv=kv, monkeypatch=monkeypatch)
    m1.layers[4] = FakeSplitLayer(4, m1.shared, kv, "swa")
    return m0, m1, ex0, ex1, wire


def split_directives(req, chunk, start, ids, blocks, parts, s0=((0, 4),)):
    """Stage 0 whole layers; stage 1: attention parts of layer 4 (the stage's first layer), then MoE(4) + [5, 8)."""
    from vllm.v1.ds41_slices import PART_ATTN, PART_MOE
    d0, _ = directives(req, chunk, start, ids, blocks, s0=s0)
    T = len(ids)
    d1 = [SliceDirective(1, req, chunk, start, T, (), blocks, 4, 5, i == 0, False, 500, PART_ATTN, i, parts)
          for i in range(parts)]
    d1.append(SliceDirective(1, req, chunk, start, T, (), blocks, 4, 8, False, True, 500, PART_MOE))
    return d0, d1


@pytest.mark.parametrize("parts", [1, 2, 3, 5])
def test_attention_parts_equal_unsliced_with_decode_steps(monkeypatch, parts) -> None:
    """Stage 1's first layer as N attention parts + a MoE part, one per step with a decode step between each:
    bit-identical to the unsliced forward (shared top-k rows of earlier parts survive via the low-row save; the
    recomputed mixes are the same)."""
    monkeypatch.setenv(sp.VERIFY_KNOB, "1")          # the executor's own unsliced re-run must agree too
    kv = torch.zeros(N_LAYERS, KV_ROWS, 16)
    m0, m1, ex0, ex1, _ = split_executors(monkeypatch, kv)
    for ex in (ex0, ex1):
        monkeypatch.setattr(ex, "_chunk_kv_views", lambda st, ids, ex=ex: [
            (f"model.layers.{i}.attn.swa", kv[i].view(-1, 16, 16),
             torch.tensor(list(ids[0])[st.start_pos // 16:(st.start_pos + st.num_tokens - 1) // 16 + 1]))
            for i in range(ex.stage_first, ex.stage_end)])
    errors: list = []
    monkeypatch.setattr(sp.logger, "error", lambda msg, *a: errors.append(msg % a))
    T, start = 40, 32
    ids = np.arange(200, 200 + T, dtype=np.int32)
    d0, d1 = split_directives("S", 0, start, ids, (tuple(range(1, 6)),), parts)
    done = step(ex0, ex1, m0, m1, [d0[0], d1[0]])
    for d in d1[1:]:
        done += step(ex0, ex1, m0, m1, [d])
    assert done == [("S", 0)] and errors == [] and ex1.stats["verified"] == 1
    got = ex1.runner.intermediate_tensors[C.PP_KEY_HIDDEN][:T].clone()
    m1r = make_model(False, True, 4, 8)
    m1r.layers[4] = FakeSplitLayer(4, m1r.shared, None, "swa")
    ref, _ = unsliced_with_md(make_model(True, False, 0, 4), m1r, torch.from_numpy(ids).long(),
                              torch.arange(start, start + T), d0[0])
    assert torch.equal(got.view(torch.int16), ref.view(torch.int16))


def test_attention_part_order_is_enforced(monkeypatch) -> None:
    m0, m1, ex0, ex1, _ = split_executors(monkeypatch)
    ids = np.arange(1, 41, dtype=np.int32)
    d0, d1 = split_directives("S", 0, 0, ids, ((1, 2, 3),), 3)
    step(ex0, ex1, m0, m1, [d0[0], d1[0]])
    with pytest.raises(RuntimeError, match="expected attention part 1/3"):
        step(ex0, ex1, m0, m1, [d1[2]])                        # part 2 before part 1
    with pytest.raises(RuntimeError, match="MoE part needs every attention part"):
        step(ex0, ex1, m0, m1, [d1[3]])                        # MoE before the last part
    whole = dataclasses.replace(d1[3], layer_part=0)
    with pytest.raises(RuntimeError, match="mid-way"):
        step(ex0, ex1, m0, m1, [whole])


def test_attention_parts_refused_before_the_replay_plan(monkeypatch) -> None:
    m1 = make_model(False, True, 4, 8)
    m1.replay = SimpleNamespace(first_layer=5)
    st = SimpleNamespace(shared=m1.shared, engram_handle=None, engram_service=None, replay_done=False,
                         replay_skip=False)
    with pytest.raises(ValueError, match="replay plan has not run"):
        m1.forward_slice(st, 5, 6, 1, 0, 2)
    with pytest.raises(ValueError, match="attention part"):
        m1.forward_slice(st, 4, 6, 1, 0, 2)                   # an attention part covers exactly one layer


def test_attention_parts_report_one_cost_sample_per_chunk(monkeypatch) -> None:
    """Adaptive feedback: a chunk's attention parts are summed into ONE (1, -1, ms, False, T * n_ctx / 1e6)
    sample; the MoE part is an ordinary slice sample."""
    monkeypatch.setenv("VLLM_DS41_SLICED_PREFILL", "1")
    monkeypatch.setenv("VLLM_DS41_SLICE_POLICY", "adaptive")
    m0, m1, ex0, ex1, _ = split_executors(monkeypatch)
    T, start = 40, 64
    d0, d1 = split_directives("F", 0, start, np.arange(1, 1 + T, dtype=np.int32), ((1, 2, 3, 4, 5, 6, 7),), 3)
    step(ex0, ex1, m0, m1, [d0[0], d1[0]])
    for d in d1[1:]:
        step(ex0, ex1, m0, m1, [d])
    slices, _ = ex1.take_feedback()
    attn = [x for x in slices if x[0] == 1 and x[1] == -1]
    assert len(attn) == 1 and attn[0][4] == pytest.approx(T * (start + T) / 1e6) and attn[0][2] > 0
    assert any(x[0] == 1 and x[1] == 4 for x in slices)        # MoE part of layer 4 + layers 5-7


def test_indexer_scan_tasks_partition_and_cap() -> None:
    """The scan's ordered tasks split into parts that cover every tile exactly once, in order; the query-tile cap
    changes the tile grid (M) but not the key tile."""
    from vllm.models.deepseek_v41.indexer import DeepseekV41Indexer
    from vllm.v1.ds41_slices import split_range

    def fake(cap):
        ix = DeepseekV41Indexer.__new__(DeepseekV41Indexer)
        object.__setattr__(ix, "_parameters", {}); object.__setattr__(ix, "_buffers", {})
        object.__setattr__(ix, "_modules", {})
        ix.__dict__.update(ratio=1, score_budget=128 << 20, qtile_cap=cap, candidate_scoring=False,
                           topo=SimpleNamespace(uses_candidates=False, is_candidate_source=True))
        return ix

    md = SimpleNamespace(num_reqs=1, query_start_loc_cpu=np.array([0, 1024]),
                         positions_cpu=np.arange(31744, 32768))
    cand = torch.zeros(1024, 4, dtype=torch.int32)
    t0, i0 = DeepseekV41Indexer.scan_tasks(fake(0), md, cand)
    t1, i1 = DeepseekV41Indexer.scan_tasks(fake(128), md, cand)
    assert [(a, b) for _, _, a, b in t0] == [(0, 1024)]            # n = 32K: one tile without the cap
    assert [(a, b) for _, _, a, b in t1] == [(a, a + 128) for a in range(0, 1024, 128)]
    assert i0 == i1                                                 # same n and key tile
    for n in (1, 2, 3, 5, 8, 11):
        got = []
        for i in range(n):
            lo, hi = split_range(len(t1), i, n)
            got += t1[lo:hi]
        assert got == t1


@pytest.mark.parametrize("n_req,parts", [(60, 2), (60, 3), (40, 2)])
def test_attention_parts_on_a_replay_layer_after_compaction(monkeypatch, n_req: int, parts: int) -> None:
    """A reindex-like layer past the replay cut (layer 6 here, replay.first_layer = 5) as attention parts on the
    chunk's kept rows: whole [4,5), [5,6) (the replay plan compacts at 5), attention parts of 6, MoE(6) + [7,8);
    decode steps between all of them; equals the unsliced forward with the same plan (compact / none)."""
    from vllm.v1.ds41_slices import PART_ATTN, PART_MOE

    m0, m1, ex0, ex1, _ = executors(monkeypatch=monkeypatch)
    m1.layers[6] = FakeSplitLayer(6, m1.shared, None, "swa")
    m1.replay = StubReplay(k=30)
    T, start = 24, 10
    ids = np.arange(1, 1 + T, dtype=np.int32)
    blocks = ((1, 2, 3),)
    d0, _ = directives("Q", 0, start, ids, blocks, s0=((0, 4),), n_req=n_req)
    mk = lambda a, b, first=False, last=False, part=0, i=0, n=1: SliceDirective(
        1, "Q", 0, start, T, (), blocks, a, b, first, last, n_req, part, i, n)
    d1 = [mk(4, 5, first=True), mk(5, 6)] + [mk(6, 7, part=PART_ATTN, i=i, n=parts) for i in range(parts)]
    d1.append(mk(6, 8, last=True, part=PART_MOE))
    inp = None
    done = step(ex0, ex1, m0, m1, [d0[0], d1[0]])
    inp_stream, inp_pre = ex0.slots[0].stream[:T].clone(), ex0.slots[0].pre[:T].clone()
    for d in d1[1:]:
        done += step(ex0, ex1, m0, m1, [d])
        if d.layer_start == 5:
            rows = ex1.slots[0].rows
    assert done == [("Q", 0)]
    got = ex1.runner.intermediate_tensors[C.PP_KEY_HIDDEN][:rows].clone()

    ref = make_model(False, True, 4, 8)
    ref.layers[6] = FakeSplitLayer(6, ref.shared, None, "swa")
    ref.replay = StubReplay(k=30)
    md = fake_md_builder(FakeRunner(True), d1[0], sp.DeviceArena(1 << 16, torch.device("cpu")), None)
    CTX.attn_metadata = md
    ref.replay.bind(1, T, np.array([0, T]), np.array([n_req]))
    plan = ref.replay.begin(md)
    s_, _, _, _, _ = ref._run_layers(inp_stream, inp_pre, torch.arange(start, start + T), 4, 8, plan=plan)
    assert rows == s_.shape[0] and torch.equal(got.view(torch.int16), s_.view(torch.int16))
