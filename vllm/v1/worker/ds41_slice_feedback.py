# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-side cost feedback for adaptive slice sizing (``VLLM_DS41_SLICE_POLICY=adaptive``; vllm/v1/ds41_slices.py).

Per stage (meaningful on TP rank 0; other TP ranks collect nothing and send zeros):

* slices: (layers, ms, first) from the slice executor's CUDA events (host clock on CPU tests), once completed;
* decode steps: fwd / post / span from the stage step timer (ds41_stage_timing) for "decode" steps; ``clean`` when
  neither this step nor the previous one carried slice directives (the span is then the undisturbed decode cycle).

The last stage puts its samples, plus stage 0's, into ``ModelRunnerOutput.ds41_slice_ms`` / ``ds41_decode_ms``.
Stage 0's samples ride its chunk handoff as one static ``f32[SLICE_FB_LEN]`` tensor (``pack`` / ``unpack``):

  [0] n_slices (<= MAX_SLICES, newest kept)  [1 .. 3*MAX_SLICES] (layers, ms, first) triples
  [-3] decode steps aggregated  [-2] sum of their fwd ms  [-1] format tag (1.0)
"""

from __future__ import annotations

from collections import deque

from vllm.v1.ds41_slices import SLICE_FB_LEN

MAX_SLICES = 9                       # 1 + 3*9 + 3 = 31 <= SLICE_FB_LEN
_MAX_KEEP = 64                       # bounded buffers if nobody drains (e.g. a non-output rank)
FORMAT_TAG = 1.0


class SliceFeedback:
    def __init__(self, stage: int) -> None:
        self.stage = int(stage)
        self.slices: deque[tuple[int, int, float, bool]] = deque(maxlen=_MAX_KEEP)
        self.decode: deque[tuple[int, float, float, bool]] = deque(maxlen=_MAX_KEEP)
        self._prev_had_slices = False
        self._dec_n = 0
        self._dec_fwd = 0.0

    # ---- samples on this stage
    def note_slice(self, layers: int, ms: float, first: bool, extra: tuple = ()) -> None:
        """(stage, layers, ms, first[, work, kind, n_reindex]): the tail is the stage-1 cost-unit breakdown
        (ds41_slices.COST_KIND_LAYERS) used by the sub-layer planner; stage 0's samples have none."""
        self.slices.append((self.stage, int(layers), float(ms), bool(first), *extra))

    def note_attn(self, ms: float, work: float, kind: int = 1) -> None:
        """Sub-layer slices (last stage): one layer's whole attention half, summed over its attention parts ->
        (stage, -1, ms, False, work, kind): kind COST_KIND_SCAN = the first layer (work T * n_ctx / 1e6),
        COST_KIND_REINDEX_ATTN = a reindex layer (work kept rows / 1024)."""
        self.slices.append((self.stage, -1, float(ms), False, float(work), int(kind)))

    def note_decode(self, kind: str, had_slices: bool, fwd: float, post: float, span: float) -> None:
        """StageStepTimer.on_step callback (steps complete in submission order)."""
        clean = not had_slices and not self._prev_had_slices
        self._prev_had_slices = had_slices
        if kind != "decode":
            return
        if self.stage == 0:
            self._dec_n += 1
            self._dec_fwd += float(fwd)
        else:
            self.decode.append((self.stage, float(fwd) + float(post), float(span), clean))

    # ---- stage 0 -> handoff -> last stage
    def pack(self) -> list[float]:
        out = [0.0] * SLICE_FB_LEN
        items = [it[:4] for it in self.slices if it[1] >= 0][-MAX_SLICES:]   # whole-layer samples (stage 0)
        self.slices.clear()
        out[0] = float(len(items))
        for i, (_, layers, ms, first) in enumerate(items):
            out[1 + 3 * i:4 + 3 * i] = [float(layers), float(ms), 1.0 if first else 0.0]
        out[-3], out[-2], out[-1] = float(self._dec_n), self._dec_fwd, FORMAT_TAG
        self._dec_n, self._dec_fwd = 0, 0.0
        return out

    def unpack(self, values, stage: int = 0) -> None:
        v = [float(x) for x in values]
        if len(v) != SLICE_FB_LEN or v[-1] != FORMAT_TAG:
            return                    # a non-TP0 rank's zeros, or an unknown format
        n = min(int(v[0]), MAX_SLICES)
        for i in range(n):
            layers, ms, first = v[1 + 3 * i:4 + 3 * i]
            self.slices.append((stage, int(layers), ms, first > 0.5))
        if v[-3] > 0:
            self.decode.append((stage, v[-2] / v[-3], 0.0, False))

    # ---- last stage -> ModelRunnerOutput
    def take(self) -> tuple[tuple, tuple]:
        s, d = tuple(self.slices), tuple(self.decode)
        self.slices.clear()
        self.decode.clear()
        return s, d
