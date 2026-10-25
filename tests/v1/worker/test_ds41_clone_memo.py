"""clone_metadata's memo must not return one KV group's clone for another group's freshly built metadata object whose
id() was reused after the first was dropped (2026-10-24 GPU gate crash: SWA layers got DS41CompressedMetadata)."""
import dataclasses
import gc

import torch

from vllm.v1.worker.ds41_sliced_prefill import DeviceArena, clone_metadata


@dataclasses.dataclass
class _SWA:
    window_slots: torch.Tensor


@dataclasses.dataclass
class _Compressed:
    slot_mapping: torch.Tensor


def test_memo_survives_id_reuse():
    arena = DeviceArena(1 << 16, torch.device("cpu"))
    memo: dict = {}
    clones = []
    for i in range(200):        # build, clone, drop: CPython reuses freed ids quickly
        md = (_Compressed if i % 2 else _SWA)(torch.full((4,), i))
        clones.append(clone_metadata(md, arena, memo))
        del md
        gc.collect()
    for i, c in enumerate(clones):
        assert isinstance(c, _Compressed if i % 2 else _SWA), f"clone {i} has the wrong type {type(c).__name__}"


def test_memo_still_dedups_the_same_object():
    arena = DeviceArena(1 << 16, torch.device("cpu"))
    memo: dict = {}
    md = _SWA(torch.zeros(4))
    assert clone_metadata(md, arena, memo) is clone_metadata(md, arena, memo)
