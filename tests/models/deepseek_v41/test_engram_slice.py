# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engram slice lane (VLLM_DS41_SLICED_PREFILL=1; hol-design.md §2a) -- CPU only.

Same CPU harness as test_engram_lookahead.py (device="cpu", CUDA calls faked; run with CUDA_VISIBLE_DEVICES="").
GPU twin: DS41_ENGRAM_SLICE_GPU=1 runs the same tests on a real CUDA device (real streams, events, H2D, expansion).
A staged prompt chunk reaches layer 1 and layer 14 in different engine steps while a decoder keeps using the step
slots in between; every slice row and every decode row is byte-compared with the synthetic tables AND with today's
single-slot path (the same chunk run as an ordinary step). Also: abort mid-slice (after layer 1, and before the gather
was ever consumed), the sampled tail through the normal path after the slices, the lane routing the module relies on
(rows_buffer / wait_rows return the lane's buffer), and the refusals. Pread and RAM-resident paths.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from vllm.models.deepseek_v41.common.contracts import EngramReqStep, EngramStepPlan

from .test_engram_lookahead import _FakeEvent, _FakeStream, check
from .test_engram_service import Oracle
from .test_engram_synth import SyntheticEngram, build_synthetic_engram, tokenizer_path

GPU = os.environ.get("DS41_ENGRAM_SLICE_GPU") == "1"


@pytest.fixture(scope="module")
def syn(tmp_path_factory: pytest.TempPathFactory) -> SyntheticEngram:
    return build_synthetic_engram(tmp_path_factory.mktemp("engram_syn_slice"))


@pytest.fixture(autouse=True)
def _cpu_only(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    import contextlib

    from vllm.models.deepseek_v41.common import engram_host as H

    monkeypatch.setenv("VLLM_DS41_ENGRAM_REQUIRE_VERIFIED", "0")
    monkeypatch.setenv("VLLM_DS41_ENGRAM_RESIDENT_LOCK", str(tmp_path / "resident.lock"))
    monkeypatch.setenv("VLLM_DS41_ENGRAM_RESIDENT_MARGIN_GIB", "0")
    monkeypatch.setenv("VLLM_DS41_SLICED_PREFILL", "1")
    if GPU:
        if not torch.cuda.is_available():
            pytest.skip("DS41_ENGRAM_SLICE_GPU=1 needs a CUDA device")
        return
    monkeypatch.setattr(H._PinnedHostBuffer, "register", lambda self: None)
    monkeypatch.setattr(H._PinnedHostBuffer, "release", lambda self: None)
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self, *a, **k: self)
    monkeypatch.setattr(torch.cuda, "Stream", _FakeStream)
    monkeypatch.setattr(torch.cuda, "Event", _FakeEvent)
    monkeypatch.setattr(torch.cuda, "stream", lambda s: contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: _FakeStream())


def make_cpu_service(syn: SyntheticEngram, tp_rank: int = 0, max_tokens: int = 256, io_threads: int = 3):
    """CPU harness, or the real device with DS41_ENGRAM_SLICE_GPU=1."""
    from vllm.models.deepseek_v41.common.engram_host import EngramHostService

    return EngramHostService(syn.hf_config, (1, 14), tp_rank, 4, str(syn.row_dir), tokenizer_path(), max_tokens,
                             torch.device("cuda" if GPU else "cpu"), io_threads=io_threads)


def _np(t: torch.Tensor) -> np.ndarray:
    if t.is_cuda:
        torch.cuda.synchronize()
    return t.detach().cpu().clone().numpy()


def step(svc, sid: int, reqs: list[EngramReqStep], order: list[str], t_pad: int, finished=frozenset()):
    from vllm.models.deepseek_v41.common.contracts import EngramBatchLayout

    svc.begin_step(EngramStepPlan(sid, tuple(reqs), frozenset(finished)))
    n_by = {r.req_id: r.num_tokens for r in reqs}
    qsl = np.cumsum([0] + [n_by[r] for r in order]).astype(np.int32)
    svc.bind_batch(EngramBatchLayout(sid, tuple(order), qsl, int(qsl[-1]), t_pad, None))
    out = {lid: _np(svc.wait_rows(lid)) for lid in svc.layers}
    svc.end_step(sid)
    return out


def slice_rows(svc, handle: int, layer_id: int, t_pad: int) -> np.ndarray:
    """What the unchanged Engram module sees inside slice_lane: rows_buffer, then wait_rows into the same buffer."""
    with svc.slice_lane(handle):
        buf = svc.rows_buffer(layer_id, t_pad)
        rows = svc.wait_rows(layer_id)
        assert rows.data_ptr() == buf.data_ptr() and rows.shape == buf.shape
        return _np(rows)


def decode(svc, orc: Oracle, sid: int, rid: str, pos: int, tok: int) -> dict[int, np.ndarray]:
    orc.set(rid, pos, [tok])
    g = step(svc, sid, [EngramReqStep(rid, pos, 1, np.array([tok], np.int32), None)], [rid], 8)
    check(orc, g, [(rid, pos, 1)], 8)
    return g


def _prompts(rng):
    pa = rng.integers(3, 129000, size=600).astype(np.int32)
    pa[300:340] = pa[20:60]                       # repeated span: dedup inside a chunk / across chunks
    pb = rng.integers(3, 129000, size=40).astype(np.int32)
    return pa, pb


def _sliced(svc, orc: Oracle, rng) -> tuple[list[dict[int, np.ndarray]], list[dict[int, np.ndarray]]]:
    """B decodes every engine step through the step slots; A's prompt goes in as two staged chunks whose layer-1 and
    layer-14 rows are consumed several steps apart, then A's tail [512, 600) and first decode take the normal path."""
    pa, pb = _prompts(rng)
    orc.set("A", 0, pa)
    orc.set("B", 0, pb)
    chunks, decodes = [], []
    g = step(svc, 0, [EngramReqStep("B", 0, 40, pb, pb)], ["B"], 40)
    check(orc, g, [("B", 0, 40)], 40)
    sid, bpos = 1, 40
    for c, (s, n, t_pad) in enumerate(((0, 256, 256), (256, 256, 264))):
        h = svc.begin_slice("A", s, n, pa if c == 0 else None, num_tokens_padded=t_pad)
        assert svc.rows_ready(h) or not svc.resident
        decodes.append(decode(svc, orc, sid, "B", bpos, 1000 + sid)); sid += 1; bpos += 1
        got = {1: slice_rows(svc, h, 1, t_pad)}               # slice k reaches layer 1
        for _ in range(3):                                    # decode steps between the two Engram layers
            decodes.append(decode(svc, orc, sid, "B", bpos, 1000 + sid)); sid += 1; bpos += 1
        got[14] = slice_rows(svc, h, 14, t_pad)                # a later slice reaches layer 14
        svc.end_slice(h)
        check(orc, got, [("A", s, n)], t_pad)
        chunks.append(got)
    g = step(svc, sid, [EngramReqStep("A", 512, 88, None, None)], ["A"], 88)     # the tail: normal path
    check(orc, g, [("A", 512, 88)], 88)
    chunks.append(g)
    orc.set("A", 600, [77])
    g = step(svc, sid + 1, [EngramReqStep("A", 600, 1, np.array([77], np.int32), None),
                            EngramReqStep("B", bpos, 1, np.array([78], np.int32), None)], ["B", "A"], 8)
    orc.set("B", bpos, [78])
    check(orc, g, [("B", bpos, 1), ("A", 600, 1)], 8)
    decodes.append(g)
    return chunks, decodes


def _single_slot(svc, orc: Oracle, rng) -> list[dict[int, np.ndarray]]:
    """Today's path for the same chunks: ordinary steps on the step slots."""
    pa, _ = _prompts(rng)
    orc.set("A", 0, pa)
    out = []
    for k, (s, n, t_pad) in enumerate(((0, 256, 256), (256, 256, 264), (512, 88, 88))):
        g = step(svc, k, [EngramReqStep("A", s, n, None, pa if k == 0 else None)], ["A"], t_pad)
        check(orc, g, [("A", s, n)], t_pad)
        out.append(g)
    return out


@pytest.mark.parametrize("resident", ["0", "1"])
def test_slice_lane_interleaved_identical_to_single_slot(syn: SyntheticEngram, monkeypatch: pytest.MonkeyPatch,
                                                         resident: str) -> None:
    monkeypatch.setenv("VLLM_DS41_ENGRAM_RESIDENT", resident)
    svc = make_cpu_service(syn, tp_rank=2, max_tokens=512)
    try:
        assert svc.slice_enabled and svc.resident == (resident == "1")
        chunks, _ = _sliced(svc, Oracle(svc, syn), np.random.default_rng(5))
        st = svc.stats()
        assert st["slices"] == 2 and st["slice_rows"] > 0
    finally:
        svc.shutdown()
    monkeypatch.setenv("VLLM_DS41_SLICED_PREFILL", "0")
    ref_svc = make_cpu_service(syn, tp_rank=2, max_tokens=512)
    try:
        assert not ref_svc.slice_enabled
        ref = _single_slot(ref_svc, Oracle(ref_svc, syn), np.random.default_rng(5))
    finally:
        ref_svc.shutdown()
    for a, b in zip(chunks, ref):
        for lid in (1, 14):
            np.testing.assert_array_equal(a[lid], b[lid])


@pytest.mark.parametrize("resident", ["0", "1"])
def test_slice_abort_mid_slice(syn: SyntheticEngram, monkeypatch: pytest.MonkeyPatch, resident: str) -> None:
    """Abort after layer 1 (preempt mid-chunk), abort before anything was consumed (gather possibly still pending),
    then the same chunk re-staged and a different request staged: all rows exact, the step slots unaffected."""
    monkeypatch.setenv("VLLM_DS41_ENGRAM_RESIDENT", resident)
    svc = make_cpu_service(syn, tp_rank=0, max_tokens=256)
    try:
        orc = Oracle(svc, syn)
        rng = np.random.default_rng(9)
        pa = rng.integers(3, 129000, size=300).astype(np.int32)
        pb = rng.integers(3, 129000, size=20).astype(np.int32)
        pc = rng.integers(3, 129000, size=120).astype(np.int32)
        orc.set("A", 0, pa)
        orc.set("C", 0, pc)
        step(svc, 0, [EngramReqStep("B", 0, 20, pb, pb)], ["B"], 32)
        orc.set("B", 0, pb)
        h = svc.begin_slice("A", 0, 200, pa)
        np.testing.assert_array_equal(slice_rows(svc, h, 1, 200), orc.rows("A", 0, 200)[:, 0])   # layer 1 only
        decode(svc, orc, 1, "B", 20, 5)
        svc.end_slice(h)                                      # preempted after layer 1
        with pytest.raises(RuntimeError, match="not staged"):
            svc.wait_rows_slice(h, 14)
        h = svc.begin_slice("A", 0, 200)                      # resumed: history kept, no prompt needed
        svc.end_slice(h)                                      # aborted before any layer consumed its rows
        h = svc.begin_slice("A", 0, 200)
        got = {lid: slice_rows(svc, h, lid, 200) for lid in (14, 1)}   # order of consumption does not matter
        svc.end_slice(h)
        check(orc, got, [("A", 0, 200)], 200)
        decode(svc, orc, 2, "B", 21, 6)
        h = svc.begin_slice("C", 0, 120, pc, num_tokens_padded=128)
        got = {lid: slice_rows(svc, h, lid, 128) for lid in (1, 14)}
        svc.end_slice(h)
        check(orc, got, [("C", 0, 120)], 128)
        # a request finished while staged: the step path drops its history, the staged slice still completes
        h = svc.begin_slice("A", 200, 100)
        svc.begin_step(EngramStepPlan(3, (EngramReqStep("B", 22, 1, np.array([7], np.int32), None),),
                                      frozenset({"A"})))
        svc.end_step(3)
        got = {lid: slice_rows(svc, h, lid, 100) for lid in (1, 14)}
        svc.end_slice(h)
        check(orc, got, [("A", 200, 100)], 100)
        with pytest.raises(KeyError, match="without token history"):
            svc.begin_slice("A", 0, 10)
        # shutdown with a staged, never-consumed slice drains it
        svc.begin_slice("C", 0, 64)
    finally:
        svc.shutdown()


def test_slice_refusals(syn: SyntheticEngram, monkeypatch: pytest.MonkeyPatch) -> None:
    svc = make_cpu_service(syn, tp_rank=1, max_tokens=128)
    try:
        p = np.random.default_rng(3).integers(3, 129000, size=150).astype(np.int32)
        with pytest.raises(ValueError, match="not prompt positions"):
            svc.begin_slice("A", 100, 60, p)                  # runs past the prompt
        with pytest.raises(ValueError, match="outside"):
            svc.begin_slice("A", 0, 129, p)                   # more than max_num_batched_tokens
        with pytest.raises(ValueError, match="outside"):
            svc.begin_slice("A", 0, 64, p, num_tokens_padded=32)
        h = svc.begin_slice("A", 0, 64, p)
        with pytest.raises(RuntimeError, match="one staged chunk"):
            svc.begin_slice("A", 64, 64)
        with svc.slice_lane(h):
            with pytest.raises(ValueError, match="staged chunk"):
                svc.rows_buffer(1, 72)                        # forward T_pad != staged T_pad
            with pytest.raises(RuntimeError, match="already active"):
                with svc.slice_lane(h):
                    pass
            with pytest.raises(RuntimeError, match="inside its own"):
                svc.end_slice(h)
        assert svc.rows_buffer(1, 72).data_ptr() == svc._dense.data_ptr(), "outside the lane: the step buffer"
        with pytest.raises(RuntimeError, match="not staged"):
            svc.wait_rows_slice(h + 1, 1)
        svc.end_slice(h)
        with pytest.raises(RuntimeError, match="not staged"):
            svc.end_slice(h)
        with pytest.raises(RuntimeError, match="not staged"):
            svc.rows_ready(h)
    finally:
        svc.shutdown()
    monkeypatch.setenv("VLLM_DS41_SLICED_PREFILL", "0")
    svc = make_cpu_service(syn, tp_rank=1, max_tokens=128)
    try:
        assert not hasattr(svc, "_slice_dense"), "the lane is not allocated without the knob"
        with pytest.raises(RuntimeError, match="VLLM_DS41_SLICED_PREFILL=1"):
            svc.begin_slice("A", 0, 8, np.arange(3, 20, dtype=np.int32))
    finally:
        svc.shutdown()


def test_slice_chunk_token_ids(syn: SyntheticEngram, monkeypatch: pytest.MonkeyPatch) -> None:
    """SliceDirective carries only the chunk's ids: history built chunk by chunk (start 0, then extensions) stages the
    same bytes as the full-prompt path; the normal tail after it works without ids; a gap is refused."""
    svc = make_cpu_service(syn, tp_rank=3, max_tokens=256)
    try:
        orc = Oracle(svc, syn)
        p = np.random.default_rng(13).integers(3, 129000, size=500).astype(np.int32)
        orc.set("A", 0, p)
        for s0, n in ((0, 200), (200, 200)):
            h = svc.begin_slice("A", s0, n, token_ids=p[s0:s0 + n])
            got = {lid: slice_rows(svc, h, lid, n) for lid in (1, 14)}
            svc.end_slice(h)
            check(orc, got, [("A", s0, n)], n)
        g = step(svc, 0, [EngramReqStep("A", 400, 100, p[400:500], None)], ["A"], 104)   # tail with its ids
        check(orc, g, [("A", 400, 100)], 104)
        svc.end_slice(svc.begin_slice("B", 0, 50, token_ids=p[:50]))
        with pytest.raises(ValueError, match="history gap"):
            svc.begin_slice("B", 100, 50, token_ids=p[100:150])
        with pytest.raises(KeyError, match="without token history"):
            svc.begin_slice("C", 64, 32, token_ids=p[64:96])
        # unseen request after a prefix-cache hit: the cached prefix ids build the history, the chunk extends it
        orc.set("D", 0, p)
        h = svc.begin_slice("D", 300, 100, prompt_token_ids=p[:300], token_ids=p[300:400])
        got = {lid: slice_rows(svc, h, lid, 100) for lid in (1, 14)}
        svc.end_slice(h)
        check(orc, got, [("D", 300, 100)], 100)
        with pytest.raises(ValueError, match="token_ids"):
            svc.begin_slice("C", 0, 32, token_ids=p[:31])
    finally:
        svc.shutdown()
