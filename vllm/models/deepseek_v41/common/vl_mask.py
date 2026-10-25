# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 vision: the image-span row mask of the current forward.

Every image-span position carries ``IMAGE_TOKEN_ID`` in ``input_ids`` (common/mm_preprocess.py). Those rows route with
the gate's ``bias_vl`` (ref:model.py Gate) and get no Engram contribution (ref:model.py Engram ``token_mask``). The
runner arms the mask from host data only on steps that schedule image tokens (``arm``); the model computes it from the
raw ``input_ids`` at the start of its forward (``begin``), the decoder replay compacts it with the rows (``compact``),
and ``end`` clears it, so text-only steps, CUDA-graph decode and the DSpark drafter never see one.
"""

from __future__ import annotations

import torch

IMAGE_TOKEN_ID = 129264

_armed = False
_mask: torch.Tensor | None = None


def arm(flag: bool) -> None:
    """Runner, per step, before the forward: True when a scheduled token lies in an image span."""
    global _armed
    _armed = bool(flag)


def begin(input_ids: torch.Tensor | None) -> None:
    global _mask
    _mask = None
    if _armed and input_ids is not None and not torch.cuda.is_current_stream_capturing():
        _mask = input_ids.reshape(-1) == IMAGE_TOKEN_ID


def compact(kept: torch.Tensor) -> None:
    global _mask
    if _mask is not None:
        _mask = _mask.index_select(0, kept)


def end() -> None:
    global _mask
    _mask = None


def active() -> bool:
    return _mask is not None


def rows(n: int) -> torch.Tensor | None:
    """The mask for an n-row input of this forward (None when no image is in flight)."""
    if _mask is None:
        return None
    if _mask.shape[0] != n:
        raise RuntimeError(f"DS41 vision mask has {_mask.shape[0]} rows, the layer input {n}")
    return _mask
