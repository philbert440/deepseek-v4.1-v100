#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU microbench: VLLM_DS41_INDEXER_FUSED vs today, the shapes of ds41_cand_prefill_bench.py (one GPU, ~2-4 min).

Per n (ratio 1 -> n = ctx) and T rows at the end of the context, candidates from layer 20's real scan + select:
  * cand  : today candidate_topk (16-row tiles) vs fused_candidate_topk [cand_scores | topk512]
  * scan  : today index_scores + apply_candidate_mask + topk_sorted (q tile cap 128) vs fused dense + mask + topk
  * l20   : the scan's score kernel alone (index_scores vs fused dense_scores) over all T rows
and the top-512 SET agreement of each fused path with today's (rows / T).
Run: CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD python scripts/ds41_indexer_fused_bench.py [--T 1024]
     [--ctx 8192,17272,65536,262144] [--drift 1,0.3]
"""

from __future__ import annotations

import argparse

import torch

from vllm.models.deepseek_v41 import indexer as ix
from vllm.models.deepseek_v41.common.candidate_blocks import apply_candidate_mask, select_candidate_blocks
from vllm.models.deepseek_v41.common.contracts import CAND_ALL, CAND_BLOCK, CAND_TOPK_BLOCKS, IDX_DIM, IDX_HEADS, IDX_TOPK
from vllm.models.deepseek_v41.sm70 import indexer_fused as fx
from vllm.models.deepseek_v41.sm70.indexer_kernels import index_scores, topk_sorted

BUDGET = 128 << 20
STORAGE = 64


def timed(fn, iters):
    fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
        a.record(); out = fn(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
    return sorted(ts)[len(ts) // 2], out


def sets_equal(a, b):
    return int((a.sort(dim=1).values == b.sort(dim=1).values).all(dim=1).sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=1024)
    ap.add_argument("--ctx", default="8192,17272,65536,262144")
    ap.add_argument("--drift", default="1,0.3")
    ap.add_argument("--iters", type=int, default=3)
    args = ap.parse_args()
    dev = torch.device("cuda")
    T = args.T
    ext = fx.load_ext()
    print(f"GPU {torch.cuda.get_device_name()} T={T} fused ext v{ext.version}")
    for drift in (float(d) for d in args.drift.split(",")):
        for n in (int(c) for c in args.ctx.split(",")):
            g = torch.Generator(device=dev).manual_seed(n)
            keys = (torch.randn(n, IDX_DIM, device=dev, generator=g) * 0.5).half()
            base = torch.randn(1, IDX_HEADS, IDX_DIM, device=dev, generator=g)
            q = ((base + drift * torch.randn(T, IDX_HEADS, IDX_DIM, device=dev, generator=g)) * 0.5).half()
            w = torch.rand(T, IDX_HEADS, device=dev, generator=g)
            ends = torch.arange(n - T + 1, n + 1, device=dev, dtype=torch.int64).clamp(min=1)
            nblk = (n + STORAGE - 1) // STORAGE
            rows = torch.zeros((nblk * STORAGE, IDX_DIM), dtype=torch.float16, device=dev)
            rows[:n] = keys
            bt = torch.arange(nblk, device=dev, dtype=torch.int32)[None, :].contiguous()
            tok2req = torch.zeros(T, dtype=torch.int32, device=dev)
            cand = torch.empty((T, CAND_TOPK_BLOCKS), dtype=torch.int32, device=dev)
            q_tile = max(1, min(T, BUDGET // (4 * n)))
            key_tile = max(1024, BUDGET // (4 * IDX_HEADS * q_tile))
            q_tile = min(q_tile, 128)
            for a in range(0, T, q_tile):
                b = min(T, a + q_tile)
                s = torch.empty((b - a, n), dtype=torch.float32, device=dev)
                index_scores(q[a:b], w[a:b], keys, s, key_tile=key_tile)
                select_candidate_blocks(s, ends[a:b], CAND_TOPK_BLOCKS, CAND_BLOCK, out=cand[a:b])
            del s
            blocks = torch.arange(CAND_TOPK_BLOCKS, device=dev, dtype=cand.dtype)[None, :]
            cx = torch.where(cand[:, :1] == CAND_ALL, blocks, cand)

            def cand_today():
                out = torch.empty((T, IDX_TOPK), dtype=torch.int32, device=dev)
                for a in range(0, T, 16):
                    out[a:a + 16] = ix.candidate_topk(q[a:a + 16], w[a:a + 16], cx[a:a + 16], ends[a:a + 16], keys,
                                                      lambda p: p)
                return out

            cand_fused = lambda: fx.fused_candidate_topk(q, w, cand, ends, rows, tok2req, bt, STORAGE)
            sc = torch.empty((min(T, fx.ROWS_PER_CALL), fx.CAND_SLOTS), dtype=torch.float32, device=dev)
            outc = torch.empty((T, IDX_TOPK), dtype=torch.int32, device=dev)
            Tc = min(T, fx.ROWS_PER_CALL)
            only_scores = lambda: ext.cand_scores(q[:Tc], w[:Tc], cand[:Tc], ends[:Tc], rows, tok2req[:Tc], bt, STORAGE,
                                                  sc[:Tc], fx.SLOTS_PER_CTA)
            only_topk = lambda: ext.topk512(sc[:Tc], fx.CAND_SLOTS, cand[:Tc], outc[:Tc])

            def scan(fused):
                out = torch.empty((T, IDX_TOPK), dtype=torch.int32, device=dev)
                for a in range(0, T, q_tile):
                    b = min(T, a + q_tile)
                    s = torch.empty((b - a, n), dtype=torch.float32, device=dev)
                    if fused:
                        fx.fused_index_scores(q[a:b], w[a:b], keys, s)
                    else:
                        index_scores(q[a:b], w[a:b], keys, s, key_tile=key_tile)
                    apply_candidate_mask(s, ends[a:b], cand[a:b], CAND_BLOCK)
                    out[a:b] = fx.fused_topk_sorted(s, ends[a:b]) if fused else topk_sorted(s, ends[a:b], IDX_TOPK)
                return out

            def l20(fused):
                for a in range(0, T, q_tile):
                    b = min(T, a + q_tile)
                    s = torch.empty((b - a, n), dtype=torch.float32, device=dev)
                    if fused:
                        fx.fused_index_scores(q[a:b], w[a:b], keys, s)
                    else:
                        index_scores(q[a:b], w[a:b], keys, s, key_tile=key_tile)

            t_ct, r_ct = timed(cand_today, args.iters)
            t_cf, r_cf = timed(cand_fused, args.iters)
            t_cs, _ = timed(only_scores, args.iters)
            t_ck, _ = timed(only_topk, args.iters)
            t_st, r_st = timed(lambda: scan(False), args.iters)
            t_sf, r_sf = timed(lambda: scan(True), args.iters)
            t_lt, _ = timed(lambda: l20(False), args.iters)
            t_lf, _ = timed(lambda: l20(True), args.iters)
            print(f"\n== drift {drift} n={n:,} T={T} ==")
            print(f"  cand  today {t_ct:7.1f} ms   fused {t_cf:6.1f} ms  [scores {t_cs:.1f} + topk {t_ck:.1f} per "
                  f"{Tc} rows]  sets == today {sets_equal(r_ct, r_cf)}/{T}")
            print(f"  scan  today {t_st:7.1f} ms   fused {t_sf:6.1f} ms  sets == today {sets_equal(r_st, r_sf)}/{T}"
                  f"  fused scan == fused cand {sets_equal(r_sf, r_cf)}/{T}")
            print(f"  l20 score kernel  today {t_lt:7.1f} ms   fused {t_lf:6.1f} ms")
            del keys, rows, q, cand, cx, sc
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
