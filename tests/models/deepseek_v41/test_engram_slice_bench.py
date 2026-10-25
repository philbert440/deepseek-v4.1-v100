# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host cost of the slice lane for a 1024-token chunk (CPU harness, synthetic tables; opt-in: DS41_ENGRAM_SLICE_BENCH=1).
begin_slice = hash + gather submit (resident: the memcpy itself); first wait = gather wait + H2D enqueue + expansion."""

from __future__ import annotations

import os
import time

import numpy as np
import pytest

from .test_engram_slice import _cpu_only, make_cpu_service, slice_rows, syn  # noqa: F401  (fixtures)

pytestmark = pytest.mark.skipif(os.environ.get("DS41_ENGRAM_SLICE_BENCH") != "1", reason="opt-in benchmark")


@pytest.mark.parametrize("resident", ["0", "1"])
def test_slice_host_cost(syn, monkeypatch: pytest.MonkeyPatch, resident: str) -> None:
    monkeypatch.setenv("VLLM_DS41_ENGRAM_RESIDENT", resident)
    svc = make_cpu_service(syn, tp_rank=0, max_tokens=1024)
    try:
        p = np.random.default_rng(0).integers(3, 129000, size=1024 * 9).astype(np.int32)
        t_begin, t_wait = [], []
        for k in range(9):
            t0 = time.perf_counter()
            h = svc.begin_slice("A", k * 1024, 1024, p if k == 0 else None)
            t1 = time.perf_counter()
            slice_rows(svc, h, 1, 1024)
            t2 = time.perf_counter()
            slice_rows(svc, h, 14, 1024)
            svc.end_slice(h)
            if k:
                t_begin.append((t1 - t0) * 1e3)
                t_wait.append((t2 - t1) * 1e3)
        print(f"\nresident={resident}: begin_slice p50 {np.median(t_begin):.2f} ms max {max(t_begin):.2f}; "
              f"first wait (+expand, CPU) p50 {np.median(t_wait):.2f} ms max {max(t_wait):.2f}")
    finally:
        svc.shutdown()
