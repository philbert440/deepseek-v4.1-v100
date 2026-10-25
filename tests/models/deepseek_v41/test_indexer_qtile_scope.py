# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The indexer scan's query-row tile cap (VLLM_DS41_INDEXER_QTILE_MAX) applies only inside a staged slice: attention
parts (tiles) and any forward under ds41_slices.slice_forward() (every forward_slice of the executor, the
SLICE_VERIFY re-run included); normal-path prefill keeps today's grid even on a sliced server."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from vllm.models.deepseek_v41 import indexer as ix
from vllm.v1 import ds41_slices

from .test_indexer_parts_cand_prefill import START, T, _indexer, setup  # noqa: F401  (fixture)


def test_cap_only_inside_staged_slices(monkeypatch, setup) -> None:
    monkeypatch.setattr(ix, "ATTN_GRAPH", False)
    seen = []
    orig = ix.DeepseekV41Indexer.scan_tasks

    def spy(self, md, cand, mode="off", capped=True):
        seen.append(capped)
        tasks, info = orig(self, md, cand, mode, capped)
        seen.append(len(tasks))
        return tasks, info

    monkeypatch.setattr(ix.DeepseekV41Indexer, "scan_tasks", spy)
    args = (torch.zeros(T, 1), torch.zeros(T, 1), torch.arange(START, START + T))
    x = _indexer(setup, "scan", qcap=8)
    x.forward(*args)                                   # normal path: today's grid (one tile of T rows)
    assert seen == [False, 1]
    seen.clear()
    with ds41_slices.slice_forward():                  # a staged slice / its verify re-run: the capped grid
        assert ds41_slices.slice_forward_active()
        x.forward(*args)
    assert seen == [True, T // 8] and not ds41_slices.slice_forward_active()
    seen.clear()
    x.forward(*args, tiles=(0, 2))                     # attention part: capped
    assert seen == [True, T // 8]


def test_executor_forward_context_enters_the_slice_scope() -> None:
    from vllm.v1.worker.ds41_sliced_prefill import SlicedPrefillExecutor

    stub = SimpleNamespace(vllm_config=None)
    with SlicedPrefillExecutor._forward_context(stub, SimpleNamespace(md_active=None, rows=0)):
        assert ds41_slices.slice_forward_active()
    assert not ds41_slices.slice_forward_active()
