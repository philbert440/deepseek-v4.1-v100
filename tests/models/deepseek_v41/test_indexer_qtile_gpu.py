# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU: does VLLM_DS41_INDEXER_QTILE_MAX change the full indexer scan's results? (layer-sliced prefill's stage-1
attention parts split the scan by query-row tiles; sliced and unsliced runs share the capped grid, so the slice gate
is exact either way -- this test tells whether the cap also matches today's uncapped numerics.)

Layer-20-shaped scan, T = 1024 rows, n = 8K / 32K / 48K keys: index_scores + select_candidate_blocks + topk_sorted
with one tile (today's grid at these n) vs 128-row tiles; compares scores, candidate blocks and top-512 bit for bit.
Run: DS41_QTILE_GPU=1 pytest tests/models/deepseek_v41/test_indexer_qtile_gpu.py -s
"""

from __future__ import annotations

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(not (torch.cuda.is_available() and os.environ.get("DS41_QTILE_GPU") == "1"),
                                reason="GPU test: set DS41_QTILE_GPU=1 on a CUDA box")


@pytest.mark.parametrize("n", [8192, 32768, 49152])
def test_scan_tiles_bit_identical(n: int) -> None:
    from vllm.models.deepseek_v41.common.candidate_blocks import select_candidate_blocks
    from vllm.models.deepseek_v41.common.contracts import CAND_BLOCK, CAND_TOPK_BLOCKS, IDX_DIM, IDX_HEADS, IDX_TOPK
    from vllm.models.deepseek_v41.sm70.indexer_kernels import index_scores, topk_sorted

    g = torch.Generator(device="cuda").manual_seed(n)
    T = 1024
    q = torch.randn(T, IDX_HEADS, IDX_DIM, device="cuda", generator=g).half()
    w = torch.randn(T, IDX_HEADS, device="cuda", generator=g)
    keys = torch.randn(n, IDX_DIM, device="cuda", generator=g).half()
    ends = torch.arange(n - T + 1, n + 1, device="cuda")
    key_tile = max(1024, (128 << 20) // (4 * IDX_HEADS * min(T, (128 << 20) // (4 * n))))

    def run(q_tile):
        sc_all, cand_all, top_all = [], [], []
        for a in range(0, T, q_tile):
            b = min(T, a + q_tile)
            scores = torch.empty((b - a, n), dtype=torch.float32, device="cuda")
            index_scores(q[a:b], w[a:b], keys, scores, key_tile=key_tile)
            sc_all.append(scores.clone())
            cand = torch.empty((b - a, CAND_TOPK_BLOCKS), dtype=torch.int32, device="cuda")
            select_candidate_blocks(scores, ends[a:b], CAND_TOPK_BLOCKS, CAND_BLOCK, out=cand)
            cand_all.append(cand)
            cols = torch.arange(n, device="cuda")
            scores.masked_fill_(cols[None, :] >= ends[a:b, None], float("-inf"))
            top_all.append(topk_sorted(scores, ends[a:b], IDX_TOPK))
        return torch.cat(sc_all), torch.cat(cand_all), torch.cat(top_all)

    full = run(min(T, (128 << 20) // (4 * n)))
    capped = run(128)
    for name, a, b in zip(("scores", "candidate blocks", "top-512"), full, capped):
        same = torch.equal(a.view(torch.int32) if a.dtype == torch.float32 else a,
                           b.view(torch.int32) if b.dtype == torch.float32 else b)
        print(f"n={n} {name}: {'bit-identical' if same else 'DIFFERENT'}")
        assert same, f"n={n}: {name} differ between the uncapped and the 128-row tile grid"


@pytest.mark.parametrize("n", [8192, 32768, 49152])
@pytest.mark.parametrize("q_tile", [128, 100, 37])
def test_fused_scan_tiles_bit_identical(n: int, q_tile: int) -> None:
    """VLLM_DS41_INDEXER_FUSED: the fused dense kernel + fused top-512 per row do not depend on the query-row tiling
    (each row is one warp's own HMMA chain over the same 64-key chunks; 4-row CTA grouping and key splits do not touch
    a row's reduction order) -- uncapped vs 128-row and unaligned 100 / 37-row tiles, bit for bit. Needs the .so."""
    from vllm.models.deepseek_v41.common.candidate_blocks import select_candidate_blocks
    from vllm.models.deepseek_v41.common.contracts import CAND_BLOCK, CAND_TOPK_BLOCKS, IDX_DIM, IDX_HEADS
    from vllm.models.deepseek_v41.sm70 import indexer_fused as fx

    g = torch.Generator(device="cuda").manual_seed(n + 7)
    T = 1024
    q = torch.randn(T, IDX_HEADS, IDX_DIM, device="cuda", generator=g).half()
    w = torch.randn(T, IDX_HEADS, device="cuda", generator=g)
    keys = torch.randn(n, IDX_DIM, device="cuda", generator=g).half()
    ends = torch.arange(n - T + 1, n + 1, device="cuda")

    def run(tile):
        sc_all, cand_all, top_all = [], [], []
        for a in range(0, T, tile):
            b = min(T, a + tile)
            scores = torch.empty((b - a, n), dtype=torch.float32, device="cuda")
            fx.fused_index_scores(q[a:b], w[a:b], keys, scores)
            sc_all.append(scores.clone())
            cand = torch.empty((b - a, CAND_TOPK_BLOCKS), dtype=torch.int32, device="cuda")
            select_candidate_blocks(scores, ends[a:b], CAND_TOPK_BLOCKS, CAND_BLOCK, out=cand)
            cand_all.append(cand)
            cols = torch.arange(n, device="cuda")
            scores.masked_fill_(cols[None, :] >= ends[a:b, None], float("-inf"))
            top_all.append(fx.fused_topk_sorted(scores, ends[a:b]))
        return torch.cat(sc_all), torch.cat(cand_all), torch.cat(top_all)

    full = run(T)
    tiled = run(q_tile)
    for name, a, b in zip(("scores", "candidate blocks", "top-512"), full, tiled):
        same = torch.equal(a.view(torch.int32) if a.dtype == torch.float32 else a,
                           b.view(torch.int32) if b.dtype == torch.float32 else b)
        print(f"fused n={n} tile={q_tile} {name}: {'bit-identical' if same else 'DIFFERENT'}")
        assert same, f"fused n={n}: {name} differ between the uncapped and the {q_tile}-row tile grid"
