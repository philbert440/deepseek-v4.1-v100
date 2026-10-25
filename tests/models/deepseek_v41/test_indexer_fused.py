# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_DS41_INDEXER_FUSED on CPU: the torch model of the fused kernels (sm70/indexer_fused.py) -- radix-select
top-512 rule (ties -> lowest position, -inf never picked, short rows), candidate slot -> position map (CAND_ALL,
-1 blocks, end bound), block-table row gather -- against the reference full scan + candidate mask. Exact inputs
(quarter-integer q / k), so the comparison tests logic, not rounding. GPU kernels: test_indexer_fused_gpu.py."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm.models.deepseek_v41 import indexer as ix
from vllm.models.deepseek_v41.common.candidate_blocks import select_candidate_blocks_torch
from vllm.models.deepseek_v41.common.contracts import CAND_ALL, CAND_BLOCK, CAND_TOPK_BLOCKS, IDX_DIM, IDX_HEADS
from vllm.models.deepseek_v41.sm70 import indexer_fused as fx


def _sort_ref(scores: torch.Tensor) -> torch.Tensor:
    """Top-512 finite by (score desc, column asc), output columns ascending, -1 pad."""
    Q, L = scores.shape
    out = torch.full((Q, 512), -1, dtype=torch.int32)
    for t in range(Q):
        s = scores[t]
        fin = (s != float("-inf")).nonzero().squeeze(1)
        order = sorted(fin.tolist(), key=lambda j: (-float(s[j]), j))[:512]
        order.sort()
        out[t, :len(order)] = torch.tensor(order, dtype=torch.int32)
    return out


@pytest.mark.parametrize("L", [16384, 3000, 512, 300, 0])
def test_topk512_rule(L: int) -> None:
    g = torch.Generator().manual_seed(L)
    Q = 6
    s = torch.randn(Q, L, generator=g)
    if L:
        s[0] = torch.round(s[0] * 4) / 4              # heavy ties (incl. at the threshold)
        s[1, ::3] = float("-inf")
        s[2] = float("-inf")                          # nothing finite
        s[3, : L // 2] = -0.0
        s[3, L // 2:] = 0.0                           # +0 / -0 ties
        s[4] = -s[4].abs()                            # all negative
    assert torch.equal(fx.topk512_emulate(s), _sort_ref(s))


def _data(T, n, seed):
    g = torch.Generator().manual_seed(seed)
    base = torch.randint(-4, 5, (1, IDX_HEADS, IDX_DIM), generator=g).float()
    q = (torch.round(base + 0.3 * torch.randint(-4, 5, (T, IDX_HEADS, IDX_DIM), generator=g).float()) / 4).half()
    keys = (torch.randint(-4, 5, (n, IDX_DIM), generator=g).float() / 4).half()
    w = torch.rand(T, IDX_HEADS, generator=g)
    return q, w, keys


def _cand(scores, ends):
    keep = select_candidate_blocks_torch(scores, ends, CAND_TOPK_BLOCKS, CAND_BLOCK)
    T, n = scores.shape
    nb = (n + CAND_BLOCK - 1) // CAND_BLOCK
    kb = torch.nn.functional.pad(keep, (0, nb * CAND_BLOCK - n)).view(T, nb, CAND_BLOCK).any(-1)
    cand = torch.full((T, CAND_TOPK_BLOCKS), -1, dtype=torch.int32)
    for t in range(T):
        ids = kb[t].nonzero().squeeze(1)
        cand[t, :ids.numel()] = ids.to(torch.int32)
        if (ends[t] + CAND_BLOCK - 1) // CAND_BLOCK <= CAND_TOPK_BLOCKS:
            cand[t, 0] = CAND_ALL
    return cand, keep


@pytest.mark.parametrize("n", [16400, 30000])
def test_candidate_model_matches_reference(n: int) -> None:
    """Fused candidate model (CAND_ALL in-kernel, block-table rows) == full scan + mask with the same tie rule."""
    T = 40
    q, w, keys = _data(T, n, seed=n)
    ends = torch.arange(n - T + 1, n + 1, dtype=torch.int64)
    if n < 16384 + T:
        ends = torch.arange(16384 - 20 + 1, 16384 + 21, dtype=torch.int64)
    ends[1] = 0
    scores = (torch.einsum("thd,nd->thn", q.float(), keys.float()).relu_() * w[:, :, None]).sum(dim=1)
    cand, keep = _cand(scores, ends)
    cols = torch.arange(n)
    ref = _sort_ref(scores.masked_fill(~keep | (cols[None, :] >= ends[:, None]), float("-inf")))
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(2))
    stored = torch.empty_like(keys)
    stored[perm] = keys
    rows_of = lambda pos: torch.where(pos >= 0, perm[pos.clamp(min=0)], torch.full_like(pos, -1))
    got = fx.fused_candidate_emulate(q, w, cand, ends, stored, rows_of)
    assert torch.equal(got, ref)
    assert bool((got[1] == -1).all())


def test_fused_gate() -> None:
    x = ix.DeepseekV41Indexer.__new__(ix.DeepseekV41Indexer)
    object.__setattr__(x, "_parameters", {}); object.__setattr__(x, "_buffers", {})
    object.__setattr__(x, "_modules", {})
    x.__dict__["fused"] = True
    assert not x.fused_active(16) and x.fused_active(17)
    x.__dict__["fused"] = False
    assert not x.fused_active(1024)


def test_missing_so_raises(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(fx, "_EXT", [])
    monkeypatch.setenv(fx.FUSED_SO_ENV, str(tmp_path / "nope.so"))
    with pytest.raises(ImportError):
        fx.load_ext()
