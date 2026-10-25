# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Layer-sliced prefill attention parts x VLLM_DS41_INDEXER_CAND_PREFILL on a reindex layer: the REAL
DeepseekV41Indexer.forward split into parts (tiles=(i, n)) publishes exactly the top-512 rows of the unsliced forward,
for every scoring mode (off / scan / auto both sides of CAND_SCAN_MAX) and for the device candidate path (graph mode).
CPU: the CUDA-only kernels are replaced by torch twins, inputs are quarter-integers so every logit is exact; the
projections are fixed tensors (they are recomputed identically by every part on the GPU)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.models.deepseek_v41 import indexer as ix
from vllm.models.deepseek_v41.common.contracts import CAND_ALL, CAND_BLOCK, CAND_TOPK_BLOCKS, IDX_DIM, IDX_HEADS

T, START = 48, 19952
N = START + T                 # visible entries of the last row (compress ratio 1)
_bmm = torch.bmm


def _mask_twin(scores, ends, cand, block_size):
    rows, width = scores.shape
    cols = torch.arange(width)
    keep = cols[None, :] < ends[:, None]
    blk = torch.zeros(rows, (width + block_size - 1) // block_size, dtype=torch.bool)
    for r in range(rows):
        if int(cand[r, 0]) == CAND_ALL:
            blk[r] = True
        else:
            ids = cand[r][cand[r] >= 0].long()
            blk[r, ids[ids < blk.shape[1]]] = True
    keep &= blk.repeat_interleave(block_size, dim=1)[:, :width]
    scores.masked_fill_(~keep, float("-inf"))


@pytest.fixture
def setup(monkeypatch):
    g = torch.Generator().manual_seed(11)
    Q = (torch.randint(-4, 5, (T, IDX_HEADS, IDX_DIM), generator=g).float() / 4).half()
    W = torch.randint(1, 5, (T, IDX_HEADS), generator=g).float() / 4
    keys = (torch.randint(-4, 5, (N, IDX_DIM), generator=g).float() / 4).half()
    ends = torch.arange(START + 1, START + T + 1, dtype=torch.int64)
    cand = torch.full((T, CAND_TOPK_BLOCKS), -1, dtype=torch.int32)
    nblk = (N + CAND_BLOCK - 1) // CAND_BLOCK
    for t in range(T):
        if t % 7 == 3:
            cand[t, 0] = CAND_ALL
        else:
            ids = torch.randperm(nblk, generator=g)[:CAND_TOPK_BLOCKS].sort().values
            cand[t, :ids.numel()] = ids.to(torch.int32)
    monkeypatch.setattr(torch, "bmm", lambda a, b, out_dtype=None: _bmm(a.float(), b.float()))
    monkeypatch.setattr(ix, "mm_fp32_full", lambda a, w: torch.zeros(a.shape[0], IDX_HEADS * IDX_DIM))
    monkeypatch.setattr(ix, "index_q_rope_qat", lambda q, pos, rope, impl=None: Q[: q.shape[0]])
    monkeypatch.setattr(ix, "mm_fp32", lambda a, w: W[: a.shape[0]])
    monkeypatch.setattr(ix, "logical_to_rows", lambda pos, req, bt, sbs: pos)
    monkeypatch.setattr(ix, "apply_candidate_mask", _mask_twin)

    def scores_twin(q, w, k, out, key_tile=16384):
        out[:, : k.shape[0]] = (torch.einsum("thd,nd->thn", q.float(), k.float()).relu_() * w[:, :, None]).sum(1)
        return out
    monkeypatch.setattr(ix, "index_scores", scores_twin)
    md = SimpleNamespace(num_actual_tokens=T, max_visible=N, num_visible=ends,
                         query_start_loc_cpu=np.array([0, T]), positions_cpu=np.arange(START, START + T), num_reqs=1,
                         token_to_req_indices=torch.zeros(T, dtype=torch.int32), block_table=None,
                         storage_block_size=1)
    return SimpleNamespace(keys=keys, cand=cand, md=md)


def _indexer(s, mode, scan_max=49152, qcap=8):
    x = ix.DeepseekV41Indexer.__new__(ix.DeepseekV41Indexer)
    for k in ("_parameters", "_buffers", "_modules"):
        object.__setattr__(x, k, {})
    shared = SimpleNamespace(topk_indices=torch.full((T, 512), -7, dtype=torch.int32),
                             candidate_blocks=s.cand.clone(), export_ik=None)
    x.__dict__.update(
        cand_prefill=mode, cand_scan_max=scan_max, candidate_scoring=True, cand_decode_rows=0, score_budget=128 << 20,
        qtile_cap=qcap, ratio=1, softmax_scale=IDX_DIM ** -0.5, layer_id=24, prefix="model.layers.24.attn.indexer",
        topo=SimpleNamespace(uses_candidates=True, is_candidate_source=False, owns_indexer=True), shared=shared,
        wq_b=None, weights_proj=None, rotary_emb=None)
    x.__dict__["_metadata"] = lambda: s.md
    x.__dict__["_key_rows"] = lambda: s.keys
    x.__dict__["_rope"] = lambda: None
    x.__dict__["_w"] = lambda lin, shape: lin
    return x


def _run(s, mode, parts, graph, **kw):
    x = _indexer(s, mode, **kw)
    if parts is None:
        x.forward(torch.zeros(T, 1), torch.zeros(T, 1), torch.arange(START, START + T))
    else:
        for i in range(parts):
            x.forward(torch.zeros(T, 1), torch.zeros(T, 1), torch.arange(START, START + T), tiles=(i, parts))
    assert torch.equal(x.shared.candidate_blocks, s.cand)            # a reindex layer never rewrites candidates
    return x.shared.topk_indices.clone()


@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize("mode,scan_max", [("off", 49152), ("scan", 49152), ("auto", 49152), ("auto", 1000)])
@pytest.mark.parametrize("parts", [2, 3, 5])
def test_parts_equal_unsliced(monkeypatch, setup, graph, mode, scan_max, parts) -> None:
    monkeypatch.setattr(ix, "ATTN_GRAPH", graph)
    whole = _run(setup, mode, None, graph, scan_max=scan_max)
    split = _run(setup, mode, parts, graph, scan_max=scan_max)
    assert not bool((whole == -7).any())                              # every row published
    assert torch.equal(split, whole), (mode, scan_max, parts, graph)


def test_modes_take_the_expected_path(monkeypatch, setup) -> None:
    """Which tasks each mode builds: scan+mask tasks are the capped query-row tiles (what the parts split), candidate
    tasks the 16-row tiles; off with CAND_ALL rows present scans (today's rule)."""
    monkeypatch.setattr(ix, "ATTN_GRAPH", False)
    kinds = {}
    for mode, scan_max in (("off", 49152), ("scan", 49152), ("auto", 49152), ("auto", 1000)):
        x = _indexer(setup, mode, scan_max=scan_max)
        tasks, _ = x.scan_tasks(setup.md, setup.cand, x.cand_prefill_mode(T))
        kinds[(mode, scan_max)] = ({k for k, *_ in tasks}, len(tasks))
    assert kinds[("off", 49152)] == ({"scan"}, T // 8)                # CAND_ALL rows present -> today's scan
    assert kinds[("scan", 49152)] == ({"scan"}, T // 8)
    assert kinds[("auto", 49152)] == ({"scan"}, T // 8)               # n < CAND_SCAN_MAX
    assert kinds[("auto", 1000)] == ({"cand"}, T // 16)               # n >= CAND_SCAN_MAX: candidates


@pytest.mark.parametrize("graph,mode", [(False, "scan"), (False, "auto"), (True, "off"), (False, "off")])
def test_each_part_writes_only_its_share(monkeypatch, setup, graph, mode) -> None:
    """Part i of n publishes a disjoint, contiguous share of the rows (the parts really split the work)."""
    monkeypatch.setattr(ix, "ATTN_GRAPH", graph)
    owner = torch.full((T,), -1)
    for i in range(3):
        x = _indexer(setup, mode, scan_max=1000 if mode == "auto" else 49152)
        x.forward(torch.zeros(T, 1), torch.zeros(T, 1), torch.arange(START, START + T), tiles=(i, 3))
        wrote = (x.shared.topk_indices != -7).any(dim=1)
        assert bool(wrote.any()) and not bool((owner[wrote] >= 0).any())
        owner[wrote] = i
    assert bool((owner >= 0).all()) and torch.equal(owner, owner.sort().values)
