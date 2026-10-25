# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Layer-sliced prefill attention parts x VLLM_DS41_INDEXER_FUSED: the REAL DeepseekV41Indexer.forward split into
2/3/5 parts publishes exactly the rows of the unsliced forward with fused on (what SLICE_VERIFY compares: fused sliced
vs fused unsliced), for CAND_PREFILL off / scan / auto (both sides of CAND_SCAN_MAX; scan = the fused dense kernel +
fused top-512) and the device candidate path. The fused kernels are replaced by their torch models
(sm70/indexer_fused.py), the rest by the twins of test_indexer_parts_cand_prefill.py; fused off -> not called."""

from __future__ import annotations

import pytest
import torch

from vllm.models.deepseek_v41 import indexer as ix
from vllm.models.deepseek_v41.common.contracts import IDX_HEADS
from vllm.models.deepseek_v41.sm70 import indexer_fused as fx

from .test_indexer_parts_cand_prefill import START, T, _run, setup  # noqa: F401  (fixture)


@pytest.fixture
def fused_twins(monkeypatch):
    calls = {"cand": 0, "dense": 0, "topk": 0}

    def cand(q, w, c, ends, keys, tok2req, bt, sbs):
        calls["cand"] += 1
        return fx.fused_candidate_emulate(q, w, c, ends, keys, lambda pos: pos)

    def dense(q, w, keys, out):
        calls["dense"] += 1
        lg = torch.einsum("thd,nd->thn", q.float(), keys.float())
        s = torch.zeros(q.shape[0], keys.shape[0])
        for h in range(IDX_HEADS):
            s = s + lg[:, h].relu() * w.float()[:, h:h + 1]
        out[:, :keys.shape[0]] = s
        return out

    def topk(scores, ends, k=512):
        calls["topk"] += 1
        return fx.topk512_emulate(scores)

    monkeypatch.setattr(fx, "fused_candidate_topk", cand)
    monkeypatch.setattr(fx, "fused_index_scores", dense)
    monkeypatch.setattr(fx, "fused_topk_sorted", topk)
    return calls


def _run_fused(monkeypatch, s, mode, parts, graph, scan_max):
    orig = ix.DeepseekV41Indexer.fused_active
    monkeypatch.setattr(ix.DeepseekV41Indexer, "fused_active", lambda self, T_: T_ > ix.DECODE_ROWS_MAX_T)
    try:
        return _run(s, mode, parts, graph, scan_max=scan_max)
    finally:
        monkeypatch.setattr(ix.DeepseekV41Indexer, "fused_active", orig)


@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize("mode,scan_max", [("off", 49152), ("scan", 49152), ("auto", 49152), ("auto", 1000)])
@pytest.mark.parametrize("parts", [2, 3, 5])
def test_fused_parts_equal_fused_unsliced(monkeypatch, setup, fused_twins, graph, mode, scan_max, parts) -> None:
    monkeypatch.setattr(ix, "ATTN_GRAPH", graph)
    whole = _run_fused(monkeypatch, setup, mode, None, graph, scan_max)
    n_whole = dict(fused_twins)
    split = _run_fused(monkeypatch, setup, mode, parts, graph, scan_max)
    assert not bool((whole == -7).any())
    assert torch.equal(split, whole), (mode, scan_max, parts, graph)
    assert sum(n_whole.values()) > 0                                   # the fused path ran
    # (fused vs today is not compared here: these quarter-integer logits are full of exact ties, which the fused
    # select breaks by lowest position and torch.topk arbitrarily -- the GPU test compares sets on real-valued data)


def test_fused_off_not_called(monkeypatch, setup, fused_twins) -> None:
    monkeypatch.setattr(ix, "ATTN_GRAPH", False)
    _run(setup, "auto", 2, False, scan_max=1000)
    assert fused_twins == {"cand": 0, "dense": 0, "topk": 0}


def test_prior_follows_fused(monkeypatch) -> None:
    from vllm.v1 import ds41_slices as sl

    monkeypatch.setenv("VLLM_DS41_INDEXER_CANDIDATE_SCORING", "1")
    monkeypatch.setenv("VLLM_DS41_INDEXER_CAND_PREFILL", "auto")
    monkeypatch.delenv("VLLM_DS41_INDEXER_CAND_SCAN_MAX", raising=False)
    monkeypatch.delenv("VLLM_DS41_SLICE_S1_REINDEX_ATTN_MS", raising=False)
    monkeypatch.delenv("VLLM_DS41_SLICE_S1_SCAN_MS_PER_MTOK2", raising=False)
    monkeypatch.setenv("VLLM_DS41_INDEXER_FUSED", "0")
    assert sl.cand_scan_max() == 49152 and sl.reindex_scores_by_scan(40000)
    off_scan, off_cand = sl.reindex_attn_prior_ms(1024, 17272), sl.reindex_attn_prior_ms(1024, 100000)
    assert sl.slice_s1_scan_ms_per_mtok2() == 1.5
    monkeypatch.setenv("VLLM_DS41_INDEXER_FUSED", "1")
    assert sl.cand_scan_max() == 36864 and not sl.reindex_scores_by_scan(40000)
    base = (1.0 - sl.MOE_SHARE) * sl.slice_s1_layer_ms()
    assert sl.reindex_attn_prior_ms(1024, 17272) == pytest.approx(base + 2.5 + 0.2 * 17.272)
    assert sl.reindex_attn_prior_ms(1024, 100000) == pytest.approx(base + 8.5)
    assert sl.reindex_attn_prior_ms(1024, 17272) < off_scan and sl.reindex_attn_prior_ms(1024, 100000) < off_cand
    assert sl.slice_s1_scan_ms_per_mtok2() == 0.4
    monkeypatch.setenv("VLLM_DS41_INDEXER_CAND_SCAN_MAX", "1000")          # explicit knob wins
    assert sl.cand_scan_max() == 1000
