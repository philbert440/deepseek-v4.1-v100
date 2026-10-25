# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RAM-resident Engram rows (VLLM_DS41_ENGRAM_RESIDENT=1) == the pread path, byte for byte -- CPU only.

Same CPU harness as test_engram_lookahead.py (device="cpu", CUDA calls faked; run with CUDA_VISIBLE_DEVICES="").
The resident service must stage exactly the bytes the pread service stages for chunked prefills mixed with decodes,
a short chunk, a finished request and an aborted step, for every TP rank, with buffered and O_DIRECT loads; and the
resident reader must equal the C++ RowReader on random rows of every owned sub-table (both bucket ends included).
Also: lookahead / admission prefetch are forced off, the memory check refuses, the post-load pread verification
catches a corrupted copy, the NUMA spec parser, and interleaved allocation.
"""

from __future__ import annotations

import contextlib

import numpy as np
import pytest
import torch

from .test_engram_lookahead import _FakeEvent, _FakeStream, _run, make_cpu_service
from .test_engram_service import Oracle
from .test_engram_synth import SyntheticEngram, build_synthetic_engram


@pytest.fixture(scope="module")
def syn(tmp_path_factory: pytest.TempPathFactory) -> SyntheticEngram:
    return build_synthetic_engram(tmp_path_factory.mktemp("engram_syn_resident"))


@pytest.fixture(autouse=True)
def _cpu_only(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from vllm.models.deepseek_v41.common import engram_host as H

    monkeypatch.setenv("VLLM_DS41_ENGRAM_REQUIRE_VERIFIED", "0")
    monkeypatch.setenv("VLLM_DS41_ENGRAM_RESIDENT_LOCK", str(tmp_path / "resident.lock"))
    monkeypatch.setenv("VLLM_DS41_ENGRAM_RESIDENT_MARGIN_GIB", "0")
    monkeypatch.setattr(H._PinnedHostBuffer, "register", lambda self: None)
    monkeypatch.setattr(H._PinnedHostBuffer, "release", lambda self: None)
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self, *a, **k: self)
    monkeypatch.setattr(torch.cuda, "Stream", _FakeStream)
    monkeypatch.setattr(torch.cuda, "Event", _FakeEvent)
    monkeypatch.setattr(torch.cuda, "stream", lambda s: contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: _FakeStream())


def _o_direct_ok(path) -> bool:
    import os

    try:
        os.close(os.open(str(path), os.O_RDONLY | os.O_DIRECT))
        return True
    except OSError:
        return False


@pytest.mark.parametrize("o_direct", ["0", "1"])
@pytest.mark.parametrize("tp_rank", [0, 1, 3])
def test_resident_service_identical_to_pread(syn: SyntheticEngram, monkeypatch: pytest.MonkeyPatch, tp_rank: int,
                                             o_direct: str) -> None:
    if o_direct == "1" and not _o_direct_ok(next(syn.row_dir.glob("*.safetensors"))):
        pytest.skip("filesystem of the synthetic shards refuses O_DIRECT")
    results, stats = {}, {}
    for res in ("0", "1"):
        monkeypatch.setenv("VLLM_DS41_ENGRAM_RESIDENT", res)
        monkeypatch.setenv("VLLM_DS41_ENGRAM_RESIDENT_O_DIRECT", o_direct)
        monkeypatch.setenv("VLLM_DS41_ENGRAM_RESIDENT_CHUNK_MIB", "1")      # many pieces per sub-table range
        monkeypatch.setenv("VLLM_DS41_ENGRAM_RESIDENT_LOAD_THREADS", "3")
        monkeypatch.setenv("VLLM_DS41_ENGRAM_LOOKAHEAD_TOKENS", "256")       # ignored when resident
        svc = make_cpu_service(syn, tp_rank=tp_rank, max_tokens=512)
        try:
            assert svc.resident == (res == "1")
            if svc.resident:
                assert svc.lookahead_tokens == 0 and not svc._ahead and not svc.prefetch_enabled
                n_rows = sum(int(svc._sub_rows[i].sum()) for i in range(len(svc.layers)))
                assert svc._resident.nbytes == n_rows * 256
            results[res] = _run(svc, Oracle(svc, syn), np.random.default_rng(11 + tp_rank))   # oracle-checked
            stats[res] = svc.stats()
        finally:
            svc.shutdown()
    for a, b in zip(results["0"], results["1"]):
        for lid in a:
            np.testing.assert_array_equal(a[lid], b[lid])
    s = stats["1"]
    assert s["misses"] == 0 and s["pages"] == 0 and s["prefetch_rows"] == 0 and s["lookahead_rows"] == 0
    assert 0 < s["hits"] <= s["unique_rows"] and s["resident_gib"] > 0   # the aborted step is never waited
    assert s["unique_rows"] == stats["0"]["unique_rows"]


def test_resident_reader_equals_row_reader(syn: SyntheticEngram, monkeypatch: pytest.MonkeyPatch) -> None:
    """Reader level: the same (file offset, scale table, scale index) jobs through both readers -> same bytes."""
    monkeypatch.setenv("VLLM_DS41_ENGRAM_RESIDENT", "1")
    svc = make_cpu_service(syn, tp_rank=2, max_tokens=512)
    from vllm.models.deepseek_v41.common.engram_host import load_engram_rows_extension

    rd = load_engram_rows_extension().RowReader(sorted({loc.weight_path for loc in svc.tables}), 2, False)
    try:
        for b in svc._scales:
            rd.add_table(b.tensor, 8)
        rng = np.random.default_rng(5)
        L, S = len(svc.layers), svc.n_sub
        per = 300
        rows = np.empty((per + 2, L, S), np.int64)
        for li, lid in enumerate(svc.layers):
            gl = svc.layout.layer_index(lid)
            for j, s in enumerate(svc.subtables):
                lo, n = svc.layout.offsets[gl][s], svc.layout.primes[gl][s]
                rows[:per, li, j] = lo + rng.integers(0, n, size=per)
                rows[per:, li, j] = (lo, lo + n - 1)
        inv, keys, file_idx, offsets, tab, sidx = svc._addresses(rows)
        n = keys.shape[0]
        args = (torch.from_numpy(file_idx), torch.from_numpy(offsets), torch.from_numpy(tab), torch.from_numpy(sidx))
        a = torch.zeros(n * 264, dtype=torch.uint8)
        b = torch.zeros(n * 264, dtype=torch.uint8)
        rd.wait(rd.submit(*args, a, 264, 256))
        t = svc._reader.submit(*args, b, 264, 256)
        assert svc._reader.done(t)
        assert svc._reader.wait(t)[:4] == [n, n, 0, 0]
        assert torch.equal(a, b)
        got = b.numpy().reshape(n, 264)[inv.reshape(-1)].reshape(rows.shape + (264,))
        for li, lid in enumerate(svc.layers):
            np.testing.assert_array_equal(got[:, li, :, :256], syn.weights[lid][rows[:, li]])
            np.testing.assert_array_equal(got[:, li, :, 256:], syn.scales[lid][rows[:, li]])
        with pytest.raises(RuntimeError, match="outside table"):
            bad = sidx.copy()
            bad[0] = svc._resident.tables[0].numel() // 256
            svc._reader.submit(args[0], args[1], args[2], torch.from_numpy(bad), b, 264, 256)
        with pytest.raises(RuntimeError, match="already-waited"):
            svc._reader.wait(t)
    finally:
        rd.shutdown()
        svc.shutdown()


def test_resident_memory_check_refuses(syn: SyntheticEngram, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_DS41_ENGRAM_RESIDENT", "1")
    monkeypatch.setenv("VLLM_DS41_ENGRAM_RESIDENT_MARGIN_GIB", "4096")
    with pytest.raises(RuntimeError, match="needs .* GiB margin"):
        make_cpu_service(syn)


def test_resident_verify_catches_corruption(syn: SyntheticEngram, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_DS41_ENGRAM_RESIDENT", "1")
    svc = make_cpu_service(syn, tp_rank=0)
    try:
        res = svc._resident
        assert res.verify(2000, seed=0) > 0
        rg = res.ranges[1]
        last = int(rg[-1, 1] + rg[-1, 2]) - 256          # last row of the last sub-table: always probed
        res.tables[1][last] ^= 0xFF
        with pytest.raises(RuntimeError, match="differs"):
            res.verify(10, seed=1)
    finally:
        svc.shutdown()


def test_resident_nodes_and_alloc() -> None:
    from vllm.models.deepseek_v41.common.engram_host import (load_engram_resident_extension, mems_allowed,
                                                             parse_resident_nodes)

    allowed = mems_allowed()
    assert parse_resident_nodes("all", 0) == allowed
    assert parse_resident_nodes("none", 3) is None
    assert parse_resident_nodes("1,0", 0) == [0, 1]
    assert parse_resident_nodes("0-1", 0) == [0, 1]
    assert parse_resident_nodes("0,1;0,1;1;none", 2) == [1]
    assert parse_resident_nodes("0,1;0,1;1;none", 3) is None
    with pytest.raises(ValueError, match="per-rank"):
        parse_resident_nodes("0;1", 2)
    with pytest.raises(ValueError, match="RESIDENT_NODES"):
        parse_resident_nodes("x", 0)
    ext = load_engram_resident_extension()
    t = ext.alloc_resident(8 << 20, allowed, True)
    t.fill_(7)
    assert int(t.sum()) == 7 * (8 << 20)
    with pytest.raises(RuntimeError, match="mbind"):
        ext.alloc_resident(1 << 20, [1000], True)
