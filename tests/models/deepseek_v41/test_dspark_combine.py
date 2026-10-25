# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSparkDeepseekV41Model._combine on CPU: TAP_SHIFT rescale + non-finite backstop (stubbed main_proj)."""

from types import SimpleNamespace

import pytest
import torch

from vllm.models.deepseek_v41.common import contracts as C
from vllm.models.deepseek_v41.sm70 import dspark_taps as dt
from vllm.models.deepseek_v41.sm70.dspark import MAIN_PROJ_INPUT_SCALE, DSparkDeepseekV41Model


@pytest.fixture(autouse=True)
def _reset():
    dt._cached[0] = None
    yield
    dt._cached[0] = None


def _model(w):
    m = object.__new__(DSparkDeepseekV41Model)
    m.main_proj = lambda x: (x.float() @ w).to(torch.float16)
    m.main_norm = SimpleNamespace(weight=torch.ones(C.HIDDEN))
    return m


def test_combine_matches_today_for_finite_rows_and_rescues_overflow(monkeypatch):
    g = torch.Generator().manual_seed(1)
    w = torch.randn(3 * C.HIDDEN, C.HIDDEN, generator=g) * 0.01
    mean = torch.randn(4, 3 * C.HIDDEN, generator=g) * 2.0
    mean[1] = 70000.0                                       # sink row
    monkeypatch.delenv(dt.TAP_SHIFT_KNOB, raising=False)
    aux0 = mean.to(torch.float16)                           # today's capture: row 1 is inf
    assert torch.isinf(aux0[1]).all()
    today = DSparkDeepseekV41Model._combine(_model(w), aux0)
    assert MAIN_PROJ_INPUT_SCALE == 2.0 ** -6
    assert not torch.isfinite(today[1]).all() and torch.isfinite(today[0]).all()
    dt._cached[0] = None
    monkeypatch.setenv(dt.TAP_SHIFT_KNOB, "6")
    aux6 = (mean * 2.0 ** -6).to(torch.float16)             # the guarded capture
    m = _model(w)
    mine = DSparkDeepseekV41Model._combine(m, aux6)
    assert torch.isfinite(mine).all()
    # finite rows: same values up to the FP16 subnormal tail (|x| < 2^-8 rounds once vs twice): <= 1 ulp on ~0.02%
    for i in (0, 2, 3):
        d = (mine[i].float() - today[i].float()).abs()
        assert d.max() <= 2.0 ** -11 and (d == 0).float().mean() > 0.999, (i, d.max())
    assert m._ds41_nonfinite_rows.tolist() == [False, False, False, False]
    # a row that is still non-finite after the projection is zeroed and flagged
    aux6[2] = float("inf")
    mine = DSparkDeepseekV41Model._combine(m, aux6)
    assert m._ds41_nonfinite_rows.tolist() == [False, False, True, False]
    assert torch.equal(mine[2], torch.zeros(C.HIDDEN, dtype=torch.float16)) and torch.isfinite(mine).all()
