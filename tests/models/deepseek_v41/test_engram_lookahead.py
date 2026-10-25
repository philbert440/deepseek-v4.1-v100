# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engram lookahead gather (VLLM_DS41_ENGRAM_LOOKAHEAD_TOKENS) and reader priorities -- CPU only.

The service runs with ``device="cpu"`` and the few CUDA calls it makes (streams, events, page-locking) replaced by
no-op fakes, so begin_step / bind_batch / wait_rows execute their real host logic without touching a GPU (run with
CUDA_VISIBLE_DEVICES=""). Every staged row is byte-compared with the synthetic tables; lookahead on and off must
stage identical bytes for chunked prefills mixed with decodes, a shorter-than-planned next chunk, a request that
finishes mid-prompt and an aborted step. The GPU twin lives in test_engram_service.py.
"""

from __future__ import annotations

import contextlib

import numpy as np
import pytest
import torch

from vllm.models.deepseek_v41.common.contracts import EngramBatchLayout, EngramReqStep, EngramStepPlan

from .test_engram_service import Oracle
from .test_engram_synth import SyntheticEngram, build_synthetic_engram, tokenizer_path


class _FakeEvent:
    def record(self, stream=None) -> None:
        pass

    def synchronize(self) -> None:
        pass


class _FakeStream:
    def __init__(self, *a, **k) -> None:
        pass

    def wait_event(self, ev) -> None:
        pass


@pytest.fixture(scope="module")
def syn(tmp_path_factory: pytest.TempPathFactory) -> SyntheticEngram:
    return build_synthetic_engram(tmp_path_factory.mktemp("engram_syn_cpu"))


@pytest.fixture(autouse=True)
def _cpu_only(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm.models.deepseek_v41.common import engram_host as H

    monkeypatch.setenv("VLLM_DS41_ENGRAM_REQUIRE_VERIFIED", "0")
    monkeypatch.setattr(H._PinnedHostBuffer, "register", lambda self: None)
    monkeypatch.setattr(H._PinnedHostBuffer, "release", lambda self: None)
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self, *a, **k: self)
    monkeypatch.setattr(torch.cuda, "Stream", _FakeStream)
    monkeypatch.setattr(torch.cuda, "Event", _FakeEvent)
    monkeypatch.setattr(torch.cuda, "stream", lambda s: contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: _FakeStream())


def make_cpu_service(syn: SyntheticEngram, tp_rank: int = 0, max_tokens: int = 256, io_threads: int = 3):
    from vllm.models.deepseek_v41.common.engram_host import EngramHostService

    return EngramHostService(syn.hf_config, (1, 14), tp_rank, 4, str(syn.row_dir), tokenizer_path(), max_tokens,
                             torch.device("cpu"), io_threads=io_threads)


def step(svc, sid: int, reqs: list[EngramReqStep], order: list[str], t_pad: int, finished=frozenset()):
    svc.begin_step(EngramStepPlan(sid, tuple(reqs), frozenset(finished)))
    n_by = {r.req_id: r.num_tokens for r in reqs}
    qsl = np.cumsum([0] + [n_by[r] for r in order]).astype(np.int32)
    svc.bind_batch(EngramBatchLayout(sid, tuple(order), qsl, int(qsl[-1]), t_pad, None))
    out = {lid: svc.wait_rows(lid).clone().numpy() for lid in svc.layers}
    svc.end_step(sid)
    return out


def check(orc: Oracle, got, segs, t_pad: int) -> None:
    exp = np.concatenate([orc.rows(r, s, n) for r, s, n in segs], axis=0)
    for li, lid in enumerate(orc.svc.layers):
        np.testing.assert_array_equal(got[lid][: exp.shape[0]], exp[:, li])
        assert not got[lid][exp.shape[0]: t_pad].any()


def test_split_and_renumber_helpers() -> None:
    from vllm.models.deepseek_v41.common.engram_host import misses_first, split_lookahead_hits

    rng = np.random.default_rng(0)
    for _ in range(50):
        keys = np.unique(rng.integers(0, 400, size=rng.integers(0, 120)))
        ahead = np.unique(rng.integers(0, 400, size=rng.integers(0, 120)))
        hit, src = split_lookahead_hits(keys, ahead)
        np.testing.assert_array_equal(hit, np.isin(keys, ahead))
        np.testing.assert_array_equal(ahead[src], keys[hit])
        if keys.size:
            inv = rng.integers(0, keys.size, size=300)
            new_inv, n_miss = misses_first(inv, hit)
            assert n_miss == int((~hit).sum())
            # slot order: misses (sorted), then hits (sorted)
            slot_keys = np.concatenate([keys[~hit], keys[hit]])
            np.testing.assert_array_equal(slot_keys[new_inv], keys[inv])


def _workload(rng):
    pa = rng.integers(3, 129000, size=700).astype(np.int32)
    pa[300:340] = pa[100:140]                      # repeated span: lookahead keys that also occur in earlier chunks
    pb = rng.integers(3, 129000, size=90).astype(np.int32)
    pc = rng.integers(3, 129000, size=400).astype(np.int32)
    return pa, pb, pc


def _run(svc, orc: Oracle, rng) -> list[dict[int, np.ndarray]]:
    pa, pb, pc = _workload(rng)
    outs = []
    for r, p in (("A", pa), ("B", pb), ("C", pc)):
        orc.set(r, 0, p)
    # 0: A chunk [0,200) + B whole prompt; C chunk [0,50)
    g = step(svc, 0, [EngramReqStep("A", 0, 200, None, pa), EngramReqStep("B", 0, 90, None, pb),
                      EngramReqStep("C", 0, 50, None, pc)], ["B", "A", "C"], 352)
    check(orc, g, [("B", 0, 90), ("A", 0, 200), ("C", 0, 50)], 352)
    outs.append(g)
    # 1: A [200,400) (lookahead hits), B decode, C [50,100)
    orc.set("B", 90, [11])
    g = step(svc, 1, [EngramReqStep("A", 200, 200, None, None), EngramReqStep("B", 90, 1, np.array([11], np.int32), None),
                      EngramReqStep("C", 50, 50, None, None)], ["B", "A", "C"], 256)
    check(orc, g, [("B", 90, 1), ("A", 200, 200), ("C", 50, 50)], 256)
    outs.append(g)
    # 2: A shorter chunk than planned [400,450); C finished mid-prompt (its lookahead rows go unused)
    orc.set("B", 91, [12])
    g = step(svc, 2, [EngramReqStep("A", 400, 50, None, None), EngramReqStep("B", 91, 1, np.array([12], np.int32), None)],
             ["B", "A"], 64, finished={"C"})
    check(orc, g, [("B", 91, 1), ("A", 400, 50)], 64)
    outs.append(g)
    # 3: aborted step: begin_step only (no bind / wait), then the next step retires it
    svc.begin_step(EngramStepPlan(3, (EngramReqStep("A", 450, 100, None, None),), frozenset()))
    # 4: A [450,700) (the rest of the prompt), B decode
    orc.set("B", 92, [13])
    g = step(svc, 4, [EngramReqStep("A", 450, 250, None, None), EngramReqStep("B", 92, 1, np.array([13], np.int32), None)],
             ["A", "B"], 256)
    check(orc, g, [("A", 450, 250), ("B", 92, 1)], 256)
    outs.append(g)
    # 5: A first decode token, B decode (no lookahead left)
    orc.set("A", 700, [14])
    orc.set("B", 93, [15])
    g = step(svc, 5, [EngramReqStep("A", 700, 1, np.array([14], np.int32), None),
                      EngramReqStep("B", 93, 1, np.array([15], np.int32), None)], ["A", "B"], 8)
    check(orc, g, [("A", 700, 1), ("B", 93, 1)], 8)
    outs.append(g)
    return outs


@pytest.mark.parametrize("io_threads", [1, 4])
def test_lookahead_identical_to_off(syn: SyntheticEngram, monkeypatch: pytest.MonkeyPatch, io_threads: int) -> None:
    results, stats = {}, {}
    for la in ("0", "256"):
        monkeypatch.setenv("VLLM_DS41_ENGRAM_LOOKAHEAD_TOKENS", la)
        svc = make_cpu_service(syn, tp_rank=1, max_tokens=512, io_threads=io_threads)
        try:
            assert svc.lookahead_tokens == int(la)
            assert svc.prefetch_enabled == (la == "0"), "lookahead turns the admission WILLNEED off by default"
            results[la] = _run(svc, Oracle(svc, syn), np.random.default_rng(7))
            stats[la] = svc.stats()
        finally:
            svc.shutdown()
    for a, b in zip(results["0"], results["256"]):
        for lid in a:
            np.testing.assert_array_equal(a[lid], b[lid])
    assert stats["0"]["lookahead_hits"] == 0 and stats["0"]["lookahead_rows"] == 0
    assert stats["256"]["lookahead_rows"] > 0
    assert stats["256"]["lookahead_hits"] > 0.5 * 12 * 400, "A's continuation chunks should be served by lookahead"
    assert stats["256"]["prefetch_rows"] == 0


def test_lookahead_knobs(syn: SyntheticEngram, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_DS41_ENGRAM_LOOKAHEAD_TOKENS", "128")
    monkeypatch.setenv("VLLM_DS41_ENGRAM_PREFETCH", "1")
    monkeypatch.setenv("VLLM_DS41_ENGRAM_PREFETCH_ORDER", "position")
    monkeypatch.setenv("VLLM_DS41_ENGRAM_GATHER_WILLNEED", "0")
    svc = make_cpu_service(syn, max_tokens=256)
    try:
        assert svc.prefetch_enabled and svc.prefetch_order == "position"
        assert not svc.gather_willneed and not svc._reader.gather_willneed()
        orc = Oracle(svc, syn)
        p = np.random.default_rng(3).integers(3, 129000, size=500).astype(np.int32)
        orc.set("Q", 0, p)
        for k, (s, n) in enumerate([(0, 128), (128, 128), (256, 200), (456, 44)]):
            g = step(svc, k, [EngramReqStep("Q", s, n, None, p if k == 0 else None)], ["Q"], 256)
            check(orc, g, [("Q", s, n)], 256)
        st = svc.stats()
        assert st["prefetch_rows"] > 0 and st["lookahead_hits"] > 0
    finally:
        svc.shutdown()
    monkeypatch.setenv("VLLM_DS41_ENGRAM_LOOKAHEAD_TOKENS", "257")
    with pytest.raises(ValueError, match="maximum"):
        make_cpu_service(syn, max_tokens=256)
    monkeypatch.setenv("VLLM_DS41_ENGRAM_LOOKAHEAD_TOKENS", "0")
    monkeypatch.setenv("VLLM_DS41_ENGRAM_PREFETCH_ORDER", "random")
    with pytest.raises(ValueError, match="PREFETCH_ORDER"):
        make_cpu_service(syn)


def test_reader_mid_priority_and_bad_priority(tmp_path) -> None:
    from vllm.models.deepseek_v41.common.engram_host import load_engram_rows_extension

    ext = load_engram_rows_extension()
    data = np.random.default_rng(1).integers(0, 256, size=1 << 20, dtype=np.uint8)
    f = tmp_path / "rows.bin"
    data.tofile(f)
    rd = ext.RowReader([str(f)], 2, False)
    try:
        n = 3000
        offs = np.random.default_rng(2).integers(0, data.size - 256, size=n).astype(np.int64)
        dsts, tickets = [], []
        for prio in (1, 0, 1):               # lookahead, step, lookahead: all complete with the right bytes
            dst = torch.zeros(n * 256, dtype=torch.uint8)
            tickets.append(rd.submit(torch.zeros(n, dtype=torch.int32), torch.from_numpy(offs),
                                     torch.full((n,), -1, dtype=torch.int32), torch.zeros(n, dtype=torch.int64),
                                     dst, 256, 256, prio))
            dsts.append(dst)
        for t, dst in zip(tickets, dsts):
            rd.wait(t)
            got = dst.numpy().reshape(n, 256)
            for i in range(0, n, 97):
                np.testing.assert_array_equal(got[i], data[offs[i]:offs[i] + 256])
        with pytest.raises(RuntimeError, match="priority"):
            rd.submit(torch.zeros(1, dtype=torch.int32), torch.zeros(1, dtype=torch.int64),
                      torch.full((1,), -1, dtype=torch.int32), torch.zeros(1, dtype=torch.int64),
                      torch.zeros(256, dtype=torch.uint8), 256, 256, 2)
    finally:
        rd.shutdown()
