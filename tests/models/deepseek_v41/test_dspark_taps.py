# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_DS41_DSPARK_TAP_SHIFT: FP16 overflow guard on the DSpark target taps (sm70/dspark_taps.py)."""

import pytest
import torch

from vllm.models.deepseek_v41.sm70 import dspark_taps as dt
from vllm.v1.spec_decode import ds41_dspark_trace as tr

HC, HID = 4, 5120


@pytest.fixture(autouse=True)
def _reset():
    dt._cached[0] = None
    dt._count[0] = None
    tr.LAST_CONTEXT.clear()
    yield
    dt._cached[0] = None
    dt._count[0] = None
    tr.LAST_CONTEXT.clear()


def _stream(rows, scale=1.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(rows, HC, HID, generator=g) * scale).to(torch.bfloat16)


def test_knob_default_off_is_original_expression(monkeypatch):
    monkeypatch.delenv(dt.TAP_SHIFT_KNOB, raising=False)
    assert dt.tap_shift() == 0
    s = _stream(3, scale=3.0)
    assert torch.equal(dt.capture_tap(s), s.float().mean(dim=1).to(torch.float16))
    assert dt.drafter_prescale() == 2.0 ** -6


@pytest.mark.parametrize("bad", ["-1", "17", "x", "6.0"])
def test_knob_strict(monkeypatch, bad):
    monkeypatch.setenv(dt.TAP_SHIFT_KNOB, bad)
    with pytest.raises(ValueError):
        dt.tap_shift()


def test_sink_row_overflows_without_shift_and_not_with(monkeypatch):
    s = _stream(4, scale=1.0)
    s[1] = 70000.0                        # attention-sink magnitude: beyond FP16's 65504, fine in bf16
    s[2, 0] = 2.0 ** 20
    plain = dt.capture_tap(s, shift=0)
    assert torch.isinf(plain[1]).all() and not torch.isfinite(plain[2]).all()
    assert torch.isfinite(plain[0]).all() and torch.isfinite(plain[3]).all()
    shifted = dt.capture_tap(s, shift=6)
    assert torch.isfinite(shifted).all()
    # finite rows: the drafter's input is bit-identical to today's path (power-of-two scaling is exact in FP16)
    today = (plain * (2.0 ** -6)).to(torch.float16)
    mine = (shifted * dt.drafter_prescale(6)).to(torch.float16)
    assert torch.equal(today[0], mine[0]) and torch.equal(today[3], mine[3])
    # the rescued rows are the true scaled mean
    ref = (s.float().mean(dim=1) * 2.0 ** -6).to(torch.float16)
    assert torch.equal(mine[1], ref[1]) and torch.equal(mine[2], ref[2])


def test_prescale_total_is_2_pow_minus_6():
    for s in (0, 3, 6, 8, 16):
        assert dt.drafter_prescale(s) * 2.0 ** -s == 2.0 ** -6


def test_zero_nonfinite_rows_off_and_on():
    x = torch.randn(5, 8, dtype=torch.float16)
    x[2, 3] = float("nan")
    x[4, 0] = float("inf")
    out, bad = dt.zero_nonfinite_rows(x, enabled=False)
    assert out is x and bad is None
    out, bad = dt.zero_nonfinite_rows(x, enabled=True)
    assert bad.tolist() == [False, False, True, False, True]
    assert torch.equal(out[2], torch.zeros(8, dtype=torch.float16)) and torch.equal(out[0], x[0])
    assert torch.isfinite(out).all()


def test_note_nonfinite_counts_and_hands_positions_to_trace(monkeypatch, tmp_path):
    bad = torch.tensor([False, True, True, False])
    pos = torch.tensor([0, 1, 2, 3])
    monkeypatch.delenv(tr.TRACE_KNOB, raising=False)
    dt.note_nonfinite(bad, pos)
    assert dt.nonfinite_rows_zeroed() == 2 and "nonfinite_pos" not in tr.LAST_CONTEXT
    monkeypatch.setenv(tr.TRACE_KNOB, str(tmp_path / "t.jsonl"))
    dt.note_nonfinite(bad, pos)
    assert dt.nonfinite_rows_zeroed() == 4 and tr.LAST_CONTEXT["nonfinite_pos"] == [1, 2]


def test_knob_on_reads_env_once(monkeypatch):
    monkeypatch.setenv(dt.TAP_SHIFT_KNOB, "6")
    assert dt.tap_shift() == 6 and dt.drafter_prescale() == 1.0
    monkeypatch.setenv(dt.TAP_SHIFT_KNOB, "8")
    assert dt.tap_shift() == 6                      # cached per process: target and drafter agree within a worker
