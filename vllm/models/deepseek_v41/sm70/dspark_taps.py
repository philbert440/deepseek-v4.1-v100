# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark target taps on SM70: FP16 capture of the bf16 HC stream (``VLLM_DS41_DSPARK_TAP_SHIFT``; default 0 = off).

The drafter's target-side input ("taps") is the HC stream entering layers 37-39 averaged over the 4 streams. The port
keeps the stream in software bf16 and hands the drafter an FP16 copy (``sm70/model.py``, ``_run_layers``). Attention
sink tokens (BOS, the chat template's role token, the first content token) carry activations far beyond FP16's 65504,
so those rows become inf at capture; the drafter's ``main_proj`` prescale (``MAIN_PROJ_INPUT_SCALE`` = 2^-6) comes after
the overflow, the row ends as NaN in ``main_x`` -> NaN K/V in the window cache (the FP8 QAT keeps NaN deliberately),
and every draft query that sees that window row yields NaN logits -> argmax 0 -> drafts [0, 0, 0, 0, 0]. Measured on
the live server (2026-10-25 overnight trace): every chat request drafts NaN until position 2 leaves the 128-token
window (anchor 131), ~40% of all steps, pos0 acceptance 0.46 overall vs 0.75 on the finite steps.

With the knob = s > 0 the capture is ``(mean * 2^-s).to(fp16)`` and the drafter multiplies by ``2^-6 * 2^s`` instead of
``2^-6``, so ``main_proj`` sees the same FP16 values for every row that was finite before (a power-of-two scaling is
exact in FP16; only elements below 2^-8 in magnitude, which land on the subnormal grid in both paths, can differ by one
ulp: measured 0.02% of elements); only the rows that overflowed change (they become finite). The
target's own forward never reads the taps: greedy output is unchanged by construction. s = 6 covers |x| < 2^22.
Rows that are still non-finite after the combine are zeroed (K = V = 0: a neutral key) and counted, instead of
poisoning the window.
"""

from __future__ import annotations

import torch

from vllm.logger import init_logger
from vllm.models.deepseek_v41 import knobs

logger = init_logger(__name__)

TAP_SHIFT_KNOB = "VLLM_DS41_DSPARK_TAP_SHIFT"
MAIN_PROJ_INPUT_SHIFT = 6                      # the drafter's main_proj prescale is 2^-6 (sm70/dspark.py)
_MAX_SHIFT = 16

_cached: list[int | None] = [None]
NONFINITE_ROWS_ZEROED = [0]                    # running count (TP rank-local), for the log line and tests


def tap_shift() -> int:
    """s of the capture prescale 2^-s; 0 = today's capture (``mean.to(fp16)``). Read once per process."""
    if _cached[0] is None:
        _cached[0] = knobs.env_int(TAP_SHIFT_KNOB, 0, minimum=0, maximum=_MAX_SHIFT)
        if _cached[0]:
            logger.info("DSpark taps: %s=%d (FP16 capture prescale 2^-%d; drafter rescales by 2^%d before main_proj)",
                        TAP_SHIFT_KNOB, _cached[0], _cached[0], _cached[0] - MAIN_PROJ_INPUT_SHIFT)
    return _cached[0]


def capture_tap(stream: torch.Tensor, shift: int | None = None) -> torch.Tensor:
    """[T, HC, HIDDEN] bf16 stream -> [T, HIDDEN] fp16 tap = mean over the HC streams (reference ``h.mean(dim=2)``),
    FP32 reduction, prescaled by 2^-shift before the FP16 cast (shift 0: the original expression, bit-identical)."""
    s = tap_shift() if shift is None else shift
    mean = stream.float().mean(dim=1)
    if s:
        mean = mean * (2.0 ** -s)               # exact (power of two) in FP32
    return mean.to(torch.float16)


def drafter_prescale(shift: int | None = None) -> float:
    """What the drafter multiplies the captured taps by before ``main_proj``: 2^-6 total once the capture's 2^-shift
    is accounted for (a power of two either way: the FP16 product is exact for finite, non-subnormal values)."""
    s = tap_shift() if shift is None else shift
    return 2.0 ** (s - MAIN_PROJ_INPUT_SHIFT)


def zero_nonfinite_rows(main_x: torch.Tensor, enabled: bool | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
    """(main_x with non-finite rows zeroed, the [T] bool row mask) when the knob is on; (main_x, None) when off.
    No host sync: the mask is consumed by the (debug) trace only."""
    if not (tap_shift() > 0 if enabled is None else enabled):
        return main_x, None
    bad = ~torch.isfinite(main_x).all(dim=-1)
    return torch.where(bad[:, None], torch.zeros_like(main_x), main_x), bad


_count: list[torch.Tensor | None] = [None]


def note_nonfinite(bad: torch.Tensor, positions: torch.Tensor) -> None:
    """Drafter-model side, once per step (knob on): accumulate the zeroed-row count on the device (no host sync) and,
    when the per-step trace (VLLM_DS41_DSPARK_TRACE) is on, hand the positions to it (debug: one sync)."""
    if _count[0] is None or _count[0].device != bad.device:
        _count[0] = torch.zeros((), dtype=torch.int64, device=bad.device)
    _count[0] += bad.sum()
    from vllm.v1.spec_decode import ds41_dspark_trace

    if ds41_dspark_trace.trace_path() is not None:
        ds41_dspark_trace.LAST_CONTEXT["nonfinite_pos"] = positions[bad].tolist()


def nonfinite_rows_zeroed() -> int:
    """Rows zeroed so far on this rank (host sync; diagnostics / tests)."""
    return 0 if _count[0] is None else int(_count[0])
