# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU: VLLM_DS41_INDEXER_FUSED_DECODE kernels (one V100, ~2 min).

Required: topk_sel == its torch model bit for bit (ties, -inf, short rows, CAND_ALL); the decode scan's top-512 and
candidate blocks, and the decode candidates' top-512, equal today's decode path per row or differ only by near-ties
(gap <= 4x the row's fused-vs-cuBLAS drift; printed); CUDA-graph capture + replay of both paths with new inputs ==
eager; determinism. Run: DS41_FDEC_GPU=1 CUDA_VISIBLE_DEVICES=0 pytest -s tests/models/deepseek_v41/test_indexer_fused_decode_gpu.py
"""

from __future__ import annotations

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(not (torch.cuda.is_available() and os.environ.get("DS41_FDEC_GPU") == "1"),
                                reason="GPU test: set DS41_FDEC_GPU=1 on a CUDA box")
S = 64
BUDGET = 128 << 20


def _paged(n, keys, g):
    nblk = (n + S - 1) // S
    perm = torch.randperm(nblk, device="cuda", generator=g).to(torch.int32)
    rows = torch.zeros((nblk * S, 128), dtype=torch.float16, device="cuda")
    j = torch.arange(n, device="cuda")
    rows[perm[j // S].long() * S + j % S] = keys
    return rows, perm[None, :].contiguous()


def _setup(n, T, seed):
    from vllm.models.deepseek_v41.common.contracts import IDX_DIM, IDX_HEADS

    g = torch.Generator(device="cuda").manual_seed(seed)
    keys = (torch.randn(n, IDX_DIM, device="cuda", generator=g) * 0.5).half()
    base = torch.randn(1, IDX_HEADS, IDX_DIM, device="cuda", generator=g)
    q = ((base + torch.randn(T, IDX_HEADS, IDX_DIM, device="cuda", generator=g)) * 0.5).half()
    w = torch.rand(T, IDX_HEADS, device="cuda", generator=g)
    ends = torch.arange(n - T + 1, n + 1, device="cuda", dtype=torch.int64)
    rows, bt = _paged(n, keys, g)
    tok2req = torch.zeros(T, dtype=torch.int32, device="cuda")
    return q, w, keys, ends, rows, bt, tok2req


def _today_scan(q, w, keys, ends):
    """Today's layer-20 decode: index_scores + select_candidate_blocks + causal mask + topk_sorted."""
    from vllm.models.deepseek_v41.common.candidate_blocks import select_candidate_blocks
    from vllm.models.deepseek_v41.sm70.indexer_kernels import index_scores, topk_sorted

    T, n = q.shape[0], keys.shape[0]
    s = torch.empty((T, n), dtype=torch.float32, device="cuda")
    index_scores(q, w, keys, s, key_tile=max(1024, BUDGET // (4 * 32 * T)))
    cand = torch.empty((T, 2048), dtype=torch.int32, device="cuda")
    select_candidate_blocks(s, ends, 2048, 8, out=cand)
    cols = torch.arange(n, device="cuda")
    s2 = s.masked_fill(cols[None, :] >= ends[:, None], float("-inf"))
    return topk_sorted(s2, ends, 512), cand, s2


def _near_ties(name, ref_s, got_s, a_cols, b_cols):
    from tests.models.deepseek_v41.test_indexer_fused_gpu import _near_ties as nt
    return nt(name, ref_s, got_s, a_cols, b_cols)


def _bufs(max_n):
    from vllm.models.deepseek_v41.sm70 import indexer_fused as fx
    return fx.decode_buffers(torch.device("cuda"), max_n)


def test_topk_sel_matches_model() -> None:
    from vllm.models.deepseek_v41.sm70 import indexer_fused as fx

    ext = fx._dec_ext()
    g = torch.Generator(device="cuda").manual_seed(1)
    for L, k in ((16384, 512), (131072, 2048), (3000, 2048), (700, 512)):
        s = torch.randn(6, L, device="cuda", generator=g)
        s[0] = torch.round(s[0] * 4) / 4
        s[1, ::3] = float("-inf")
        s[2] = float("-inf")
        s[3] = 0.0
        s[3, ::2] = -0.0
        lens = torch.tensor([L * 8, L * 4, L * 8, 5000, 8 * 1000, 17], device="cuda", dtype=torch.int64)
        out = torch.empty((6, k), dtype=torch.int32, device="cuda")
        ext.topk_sel(s, L, lens, 8, None, out, k, 2048 if k == 2048 else 0)
        lr = ((lens + 7) // 8).clamp(max=L).cpu()
        ref = fx.topk_sel_emulate(s.cpu(), k, lr)
        if k == 2048:
            ref[lr <= 2048, 0] = -2
        assert torch.equal(out.cpu(), ref), (L, k)


@pytest.mark.parametrize("n,T", [(8192, 1), (17272, 6), (131072, 6), (131072, 16), (1 << 20, 2)])
def test_decode_scan_vs_today(n, T) -> None:
    from vllm.models.deepseek_v41.sm70 import indexer_fused as fx

    q, w, keys, ends, rows, bt, tok2req = _setup(n, T, seed=n + T)
    top_t, cand_t, s_t = _today_scan(q, w, keys, ends)
    top_f = torch.empty((T, 512), dtype=torch.int32, device="cuda")
    cand_f = torch.empty((T, 2048), dtype=torch.int32, device="cuda")
    bufs = _bufs(1 << 20)
    fx.fused_decode_scan(q, w, ends, rows, tok2req, bt, S, top_f, cand_f, bufs)
    again = torch.empty_like(top_f)
    fx.fused_decode_scan(q, w, ends, rows, tok2req, bt, S, again, None, bufs)
    assert torch.equal(top_f, again), "not deterministic"
    s_f = torch.empty_like(s_t)
    fx.fused_index_scores(q, w, keys, s_f)
    cols = torch.arange(n, device="cuda")
    s_f.masked_fill_(cols[None, :] >= ends[:, None], float("-inf"))
    print(f"\nn={n:,} T={T}")
    _near_ties("decode scan top-512", s_t, s_f, top_t, top_f)
    nb = (n + 7) // 8
    blk = lambda s: torch.nn.functional.pad(s, (0, nb * 8 - n), value=float("-inf")).view(T, nb, 8).amax(-1)
    if int(cand_t[0, 0]) != -2:
        _near_ties("decode scan blocks", blk(s_t), blk(s_f), cand_t, cand_f)
    else:
        assert bool((cand_f[:, 0] == -2).all())


@pytest.mark.parametrize("n,T", [(17272, 6), (131072, 12), (1 << 20, 1)])
def test_decode_candidates_vs_today(n, T) -> None:
    from vllm.models.deepseek_v41 import indexer as ix
    from vllm.models.deepseek_v41.sm70 import indexer_fused as fx
    from vllm.models.deepseek_v41.sm70.sparse_kernels import logical_to_rows

    q, w, keys, ends, rows, bt, tok2req = _setup(n, T, seed=n * 3 + T)
    _, cand, _ = _today_scan(q, w, keys, ends)
    blocks = torch.arange(2048, device="cuda", dtype=cand.dtype)[None, :]
    cx = torch.where(cand[:, :1] == -2, blocks, cand)
    today = torch.cat([ix.candidate_topk(q[a:a + 2], w[a:a + 2], cx[a:a + 2], ends[a:a + 2], rows,
                                         lambda pos, a=a: logical_to_rows(pos, tok2req[a:a + 2], bt, S))
                       for a in range(0, T, 2)])
    got = torch.empty((T, 512), dtype=torch.int32, device="cuda")
    fx.fused_decode_candidates(q, w, cand, ends, rows, tok2req, bt, S, got, _bufs(1 << 20))
    s_t = torch.empty((T, n), dtype=torch.float32, device="cuda")
    s_f = torch.empty_like(s_t)
    from vllm.models.deepseek_v41.sm70.indexer_kernels import index_scores
    index_scores(q, w, keys, s_t, key_tile=4096)
    fx.fused_index_scores(q, w, keys, s_f)
    print(f"\nn={n:,} T={T}")
    _near_ties("decode candidates top-512", s_t, s_f, today, got)


def test_graph_capture_replay_equals_eager() -> None:
    from vllm.models.deepseek_v41.sm70 import indexer_fused as fx

    n, T = 131072, 6
    q, w, keys, ends, rows, bt, tok2req = _setup(n, T, seed=9)
    bufs = _bufs(1 << 20)
    top = torch.empty((T, 512), dtype=torch.int32, device="cuda")
    cand = torch.empty((T, 2048), dtype=torch.int32, device="cuda")
    top2 = torch.empty_like(top)
    fx.fused_decode_scan(q, w, ends, rows, tok2req, bt, S, top, cand, bufs)       # warm-up / lazy init outside
    fx.fused_decode_candidates(q, w, cand, ends, rows, tok2req, bt, S, top2, bufs)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        g.capture_begin()
        fx.fused_decode_scan(q, w, ends, rows, tok2req, bt, S, top, cand, bufs)
        fx.fused_decode_candidates(q, w, cand, ends, rows, tok2req, bt, S, top2, bufs)
        g.capture_end()
    torch.cuda.current_stream().wait_stream(s)
    gen = torch.Generator(device="cuda").manual_seed(10)
    q.copy_((torch.randn(q.shape, device="cuda", generator=gen) * 0.5).half())
    ends.copy_(ends - 5000)
    g.replay()
    torch.cuda.synchronize()
    e_top, e_cand, e_top2 = torch.empty_like(top), torch.empty_like(cand), torch.empty_like(top2)
    fx.fused_decode_scan(q, w, ends, rows, tok2req, bt, S, e_top, e_cand, bufs)
    fx.fused_decode_candidates(q, w, e_cand, ends, rows, tok2req, bt, S, e_top2, bufs)
    assert torch.equal(top, e_top) and torch.equal(cand, e_cand) and torch.equal(top2, e_top2)
    print(f"\ndecode static buffers: {fx.decode_buffer_bytes() / 2**20:.1f} MiB")
