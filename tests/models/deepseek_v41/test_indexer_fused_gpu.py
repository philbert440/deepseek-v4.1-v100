# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU: VLLM_DS41_INDEXER_FUSED kernels vs today's path (one V100, ~2 min).

Required: per row, top-512 SETS (and layer-20 candidate blocks) identical to today's (candidate_topk; index_scores +
mask + topk_sorted) OR a near-tie: every differing row's swapped elements are within NEAR_TIE_X x the row's own
fused-vs-cuBLAS score drift of each other under today's scores (printed with the gap); the select kernel alone on
today's scores must give today's sets exactly; the top-k kernel == its torch model bit for bit; run-to-run
determinism. Reported: score drift vs cuBLAS (max |d|, max rel, % bit-identical) and order drift (rows whose top-512
ranked by score differs, using each path's own scores).
Run: DS41_FUSED_GPU=1 CUDA_VISIBLE_DEVICES=0 pytest -s tests/models/deepseek_v41/test_indexer_fused_gpu.py
(needs the prebuilt .so: scripts/build_ds41_indexer_fused.py)
"""

from __future__ import annotations

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(not (torch.cuda.is_available() and os.environ.get("DS41_FUSED_GPU") == "1"),
                                reason="GPU test: set DS41_FUSED_GPU=1 on a CUDA box")

BUDGET = 128 << 20
NEAR_TIE_X = 4.0          # a swap is a near-tie when its score gap <= 4 x the row's max |fused - cuBLAS| drift


def _near_ties(name, ref_s, got_s, ref_cols, got_cols) -> int:
    """Rows whose selected column sets differ; asserts each is a near-tie and prints the boundary gaps. ref_s / got_s
    [T, L] scores of the two paths, *_cols [T, K] selected columns (-1 ignored)."""
    T = ref_s.shape[0]
    bad = []
    n_diff = 0
    for t in range(T):
        a = set(ref_cols[t][ref_cols[t] >= 0].tolist()); b = set(got_cols[t][got_cols[t] >= 0].tolist())
        if a == b:
            continue
        n_diff += 1
        only_r, only_g = sorted(a - b), sorted(b - a)
        fin = torch.isfinite(ref_s[t]) & torch.isfinite(got_s[t])
        drift = float((ref_s[t][fin] - got_s[t][fin]).abs().max())
        rs = ref_s[t]
        vals = torch.tensor([float(rs[j]) for j in only_r + only_g])
        gap = float(vals.max() - vals.min()) if len(vals) else 0.0
        tol = NEAR_TIE_X * max(drift, 1e-30)
        print(f"  {name} row {t}: {len(only_r)} swapped, boundary gap {gap:.3e} (today's scores, "
              f"{gap / max(abs(float(vals.abs().max())), 1e-30):.2e} rel), row drift {drift:.3e} -> "
              f"{'near-tie' if gap <= tol else 'NOT A NEAR-TIE'}")
        if gap > tol:
            bad.append(t)
    print(f"  {name}: {T - n_diff}/{T} rows identical, {n_diff - len(bad)} near-ties, {len(bad)} real differences")
    assert not bad, f"{name}: rows {bad[:8]} differ beyond rounding"
    return n_diff


def _setup(n: int, T: int, drift: float, seed: int, storage: int = 64):
    from vllm.models.deepseek_v41.common.candidate_blocks import select_candidate_blocks
    from vllm.models.deepseek_v41.common.contracts import CAND_BLOCK, CAND_TOPK_BLOCKS, IDX_DIM, IDX_HEADS
    from vllm.models.deepseek_v41.sm70.indexer_kernels import index_scores

    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(seed)
    keys = (torch.randn(n, IDX_DIM, device=dev, generator=g) * 0.5).half()
    base = torch.randn(1, IDX_HEADS, IDX_DIM, device=dev, generator=g)
    q = ((base + drift * torch.randn(T, IDX_HEADS, IDX_DIM, device=dev, generator=g)) * 0.5).half()
    w = torch.rand(T, IDX_HEADS, device=dev, generator=g)
    ends = torch.arange(n - T + 1, n + 1, device=dev, dtype=torch.int64).clamp(min=1)
    cand = torch.empty((T, CAND_TOPK_BLOCKS), dtype=torch.int32, device=dev)
    qt = max(1, min(T, BUDGET // (4 * n), 128))
    for a in range(0, T, qt):
        b = min(T, a + qt)
        s = torch.empty((b - a, n), dtype=torch.float32, device=dev)
        index_scores(q[a:b], w[a:b], keys, s, key_tile=max(1024, BUDGET // (4 * IDX_HEADS * (b - a))))
        select_candidate_blocks(s, ends[a:b], CAND_TOPK_BLOCKS, CAND_BLOCK, out=cand[a:b])
    # a shuffled paged cache: logical block i of the request lives at physical block perm[i]
    nblk = (n + storage - 1) // storage
    perm = torch.randperm(nblk, device=dev, generator=g).to(torch.int32)
    rows = torch.zeros((nblk * storage, IDX_DIM), dtype=torch.float16, device=dev)
    j = torch.arange(n, device=dev)
    rows[perm[j // storage].long() * storage + j % storage] = keys
    bt = perm[None, :].contiguous()
    tok2req = torch.zeros(T, dtype=torch.int32, device=dev)
    return q, w, keys, ends, cand, rows, bt, tok2req, storage


def _sets_equal(a: torch.Tensor, b: torch.Tensor) -> int:
    return int((a.sort(dim=1).values == b.sort(dim=1).values).all(dim=1).sum())


def _drift(name, ref: torch.Tensor, got: torch.Tensor) -> None:
    fin = torch.isfinite(ref) & torch.isfinite(got)
    assert torch.equal(torch.isfinite(ref), torch.isfinite(got)), f"{name}: -inf pattern differs"
    d = (ref[fin] - got[fin]).abs()
    rel = d / ref[fin].abs().clamp(min=1e-30)
    same = float((ref[fin].view(torch.int32) == got[fin].view(torch.int32)).float().mean())
    print(f"  {name} score drift: max |d| {float(d.max()):.3e}, max rel {float(rel.max()):.3e}, "
          f"bit-identical {100 * same:.3f}%")


def _order_drift(name, ref_s, got_s, ref_pos, got_pos) -> None:
    """rows whose top-512 ranked by score (desc, ties by position) differs between the paths."""
    T = ref_s.shape[0]
    diff = 0
    for t in range(T):
        def ranked(s, p):
            v = s[t].gather(0, p.long()) if p.numel() else s[t][:0]
            o = torch.argsort(v, descending=True, stable=True)
            return p[o]
        if not torch.equal(ranked(ref_s, ref_pos[t][ref_pos[t] >= 0]), ranked(got_s, got_pos[t][got_pos[t] >= 0])):
            diff += 1
    print(f"  {name} order drift: {diff}/{T} rows rank their top-512 differently")


def test_topk_kernel_matches_model() -> None:
    from vllm.models.deepseek_v41.sm70 import indexer_fused as fx

    ext = fx.load_ext()
    g = torch.Generator(device="cuda").manual_seed(0)
    for L in (16384, 3000, 512, 200):
        s = torch.randn(8, L, device="cuda", generator=g)
        s[0] = torch.round(s[0] * 4) / 4
        s[1, ::3] = float("-inf")
        s[2] = float("-inf")
        s[3] = 0.0
        s[3, ::2] = -0.0
        out = torch.empty((8, 512), dtype=torch.int32, device="cuda")
        ext.topk512(s, L, None, out)
        assert torch.equal(out.cpu(), fx.topk512_emulate(s.cpu())), f"L={L}"


@pytest.mark.parametrize("n,drift", [(17272, 1.0), (65536, 1.0), (65536, 0.3), (262144, 1.0)])
def test_candidate_sets_vs_today(n: int, drift: float) -> None:
    from vllm.models.deepseek_v41 import indexer as ix
    from vllm.models.deepseek_v41.common.contracts import CAND_ALL, CAND_TOPK_BLOCKS
    from vllm.models.deepseek_v41.sm70 import indexer_fused as fx
    from vllm.models.deepseek_v41.sm70.sparse_kernels import logical_to_rows

    T = 256
    q, w, keys, ends, cand, rows, bt, tok2req, S = _setup(n, T, drift, seed=n)
    blocks = torch.arange(CAND_TOPK_BLOCKS, device="cuda", dtype=cand.dtype)[None, :]
    cx = torch.where(cand[:, :1] == CAND_ALL, blocks, cand)
    rows_of = lambda a, b: (lambda pos: logical_to_rows(pos, tok2req[a:b], bt, S))
    today = torch.cat([ix.candidate_topk(q[a:a + 16], w[a:a + 16], cx[a:a + 16], ends[a:a + 16], rows,
                                         rows_of(a, a + 16)) for a in range(0, T, 16)])
    fused = fx.fused_candidate_topk(q, w, cand, ends, rows, tok2req, bt, S)
    again = fx.fused_candidate_topk(q, w, cand, ends, rows, tok2req, bt, S)
    assert torch.equal(fused, again), "fused candidate path is not deterministic"
    # today's slot scores (candidate_topk's own ops) vs the fused kernel's
    pos = fx.cand_positions(cand, ends)
    k = rows.index_select(0, logical_to_rows(pos, tok2req, bt, S).clamp(min=0).view(-1)).view(T, -1, 128)
    ref_s = torch.bmm(q, k.transpose(1, 2), out_dtype=torch.float32).relu_().mul_(w[:, :, None]).sum(1)
    ref_s.masked_fill_(pos < 0, float("-inf"))
    got_s = torch.empty((T, fx.CAND_SLOTS), dtype=torch.float32, device="cuda")
    fx.load_ext().cand_scores(q, w, cand, ends, rows, tok2req, bt, S, got_s, fx.SLOTS_PER_CTA)
    print(f"\nn={n:,} drift={drift} T={T}")
    _drift("cand", ref_s, got_s)
    # order drift on the slot scores: the picked slots ranked by each path's own scores
    T_, P = pos.shape
    slot = torch.full((T_, int(pos.max()) + 2), -1, dtype=torch.int64, device="cuda")
    ok = pos >= 0
    slot.scatter_(1, torch.where(ok, pos, torch.full_like(pos, slot.shape[1] - 1)),
                  torch.arange(P, device="cuda").expand(T_, -1))
    to_slot = lambda p: torch.where(p >= 0, slot.gather(1, p.clamp(min=0).long()), torch.full_like(p.long(), -1))
    _order_drift("cand", ref_s, got_s, to_slot(today), to_slot(fused))
    _near_ties("cand", ref_s, got_s, to_slot(today), to_slot(fused))


@pytest.mark.parametrize("n", [8192, 17272, 49152])
def test_dense_scan_vs_today(n: int) -> None:
    from vllm.models.deepseek_v41.common.candidate_blocks import apply_candidate_mask, select_candidate_blocks
    from vllm.models.deepseek_v41.common.contracts import CAND_BLOCK, CAND_TOPK_BLOCKS, IDX_TOPK
    from vllm.models.deepseek_v41.sm70 import indexer_fused as fx
    from vllm.models.deepseek_v41.sm70.indexer_kernels import index_scores, topk_sorted

    T = 256
    q, w, keys, ends, cand, *_ = _setup(n, T, 1.0, seed=n + 1)
    ref = torch.empty((T, n), dtype=torch.float32, device="cuda")
    index_scores(q, w, keys, ref, key_tile=max(1024, BUDGET // (4 * 32 * T)))
    got = torch.empty_like(ref)
    fx.fused_index_scores(q, w, keys, got)
    again = torch.empty_like(ref)
    fx.fused_index_scores(q, w, keys, again)
    assert torch.equal(got.view(torch.int32), again.view(torch.int32)), "dense kernel is not deterministic"
    print(f"\nn={n:,} T={T}")
    _drift("dense", ref, got)
    # layer 20: candidate blocks from each path's scores
    c_ref = torch.empty((T, CAND_TOPK_BLOCKS), dtype=torch.int32, device="cuda")
    c_got = torch.empty_like(c_ref)
    select_candidate_blocks(ref, ends, CAND_TOPK_BLOCKS, CAND_BLOCK, out=c_ref)
    select_candidate_blocks(got, ends, CAND_TOPK_BLOCKS, CAND_BLOCK, out=c_got)
    nb = (n + CAND_BLOCK - 1) // CAND_BLOCK
    cols_ = torch.arange(n, device="cuda")
    def block_scores(s):
        m = s.masked_fill(cols_[None, :] >= ends[:, None], float("-inf"))
        m = torch.nn.functional.pad(m, (0, nb * CAND_BLOCK - n), value=float("-inf"))
        return m.view(T, nb, CAND_BLOCK).amax(-1)
    _near_ties("layer-20 candidate blocks", block_scores(ref), block_scores(got), c_ref, c_got)
    cols = torch.arange(n, device="cuda")
    for name, mask in (("layer 20", None), ("scan+mask", cand)):
        r, gs = ref.clone(), got.clone()
        if mask is None:
            r.masked_fill_(cols[None, :] >= ends[:, None], float("-inf"))
            gs.masked_fill_(cols[None, :] >= ends[:, None], float("-inf"))
        else:
            apply_candidate_mask(r, ends, mask, CAND_BLOCK)
            apply_candidate_mask(gs, ends, mask, CAND_BLOCK)
        today = topk_sorted(r, ends, IDX_TOPK)
        fused = fx.fused_topk_sorted(gs, ends)
        cross = fx.fused_topk_sorted(r, ends)                     # same scores: the select kernel alone
        same_sel = _sets_equal(today, cross)
        _order_drift(name, r, gs, today, fused)
        print(f"  {name}: select kernel alone on today's scores: {same_sel}/{T} identical sets")
        assert same_sel == T
        _near_ties(name, r, gs, today, fused)
