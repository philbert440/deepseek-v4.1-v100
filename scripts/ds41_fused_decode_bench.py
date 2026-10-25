#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU microbench, VLLM_DS41_INDEXER_FUSED_DECODE vs today's decode indexer (one GPU, ~1-2 min).
Per (n, T): layer 20 (today: index_scores + select_candidate_blocks + mask + topk_sorted, eager; fused: block_max +
top-2048 blocks + rescore + top-512, also as a CUDA-graph replay) and one reindex layer (today: candidate_topk in
2-row tiles = VLLM_DS41_ATTN_CAND_DECODE_ROWS=2, graph replay; fused: decode candidates, graph replay); ms per call.
Run: CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD python scripts/ds41_fused_decode_bench.py [--ctx 8192,131072,1048576] [--T 1,2,6,12,16]"""

from __future__ import annotations

import argparse

import torch

from vllm.models.deepseek_v41 import indexer as ix
from vllm.models.deepseek_v41.common.candidate_blocks import select_candidate_blocks
from vllm.models.deepseek_v41.sm70 import indexer_fused as fx
from vllm.models.deepseek_v41.sm70.indexer_kernels import index_scores, topk_sorted
from vllm.models.deepseek_v41.sm70.sparse_kernels import logical_to_rows

S = 64


def timed(fn, iters=20):
    fn(); torch.cuda.synchronize()
    a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(iters):
        fn()
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) / iters


def graphed(fn):
    fn(); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        g.capture_begin(); fn(); g.capture_end()
    torch.cuda.current_stream().wait_stream(s)
    return g.replay


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", default="8192,131072,1048576")
    ap.add_argument("--T", default="1,2,6,12,16")
    a = ap.parse_args()
    bufs = fx.decode_buffers(torch.device("cuda"), 1 << 20)
    print(f"GPU {torch.cuda.get_device_name()} decode buffers {fx.decode_buffer_bytes() / 2**20:.1f} MiB")
    for n in (int(x) for x in a.ctx.split(",")):
        g = torch.Generator(device="cuda").manual_seed(n)
        keys = (torch.randn(n, 128, device="cuda", generator=g) * 0.5).half()
        nblk = (n + S - 1) // S
        rows = torch.zeros((nblk * S, 128), dtype=torch.float16, device="cuda"); rows[:n] = keys
        bt = torch.arange(nblk, device="cuda", dtype=torch.int32)[None, :].contiguous()
        for T in (int(x) for x in a.T.split(",")):
            q = (torch.randn(T, 32, 128, device="cuda", generator=g) * 0.5).half()
            w = torch.rand(T, 32, device="cuda", generator=g)
            ends = torch.arange(n - T + 1, n + 1, device="cuda", dtype=torch.int64)
            tok2req = torch.zeros(T, dtype=torch.int32, device="cuda")
            cols = torch.arange(n, device="cuda")

            def l20_today():
                j = torch.arange(n, device="cuda")[None, :]
                kr = rows.index_select(0, logical_to_rows(j, tok2req[:1], bt, S)[0])
                s = torch.empty((T, n), dtype=torch.float32, device="cuda")
                index_scores(q, w, kr, s, key_tile=max(1024, (128 << 20) // (4 * 32 * T)))
                c = torch.empty((T, 2048), dtype=torch.int32, device="cuda")
                select_candidate_blocks(s, ends, 2048, 8, out=c)
                s.masked_fill_(cols[None, :] >= ends[:, None], float("-inf"))
                return topk_sorted(s, ends, 512), c
            top, cand = l20_today()
            out = torch.empty((T, 512), dtype=torch.int32, device="cuda")
            cf = torch.empty((T, 2048), dtype=torch.int32, device="cuda")
            l20_fused = lambda: fx.fused_decode_scan(q, w, ends, rows, tok2req, bt, S, out, cf, bufs)
            blocks = torch.arange(2048, device="cuda", dtype=cand.dtype)[None, :]
            cx = torch.where(cand[:, :1] == -2, blocks, cand)
            o2 = torch.empty((T, 512), dtype=torch.int32, device="cuda")

            def re_today():
                for s0 in range(0, T, 2):
                    o2[s0:s0 + 2] = ix.candidate_topk(q[s0:s0 + 2], w[s0:s0 + 2], cx[s0:s0 + 2], ends[s0:s0 + 2], rows,
                                                      lambda pos, s0=s0: logical_to_rows(pos, tok2req[s0:s0 + 2], bt, S))
            o3 = torch.empty_like(o2)
            re_fused = lambda: fx.fused_decode_candidates(q, w, cand, ends, rows, tok2req, bt, S, o3, bufs)
            t_l20 = timed(l20_today, 5)
            t_l20f = timed(l20_fused); t_l20g = timed(graphed(l20_fused))
            t_ret = timed(graphed(re_today)); t_ref = timed(graphed(re_fused))
            same = int((top.sort(1).values == out.sort(1).values).all(1).sum())
            print(f"n={n:>8,} T={T:2d} | layer20 today {t_l20:6.2f} ms (eager)  fused {t_l20f:5.2f} eager / {t_l20g:5.2f} "
                  f"graph  sets==today {same}/{T} | reindex today {t_ret:5.2f} graph  fused {t_ref:5.2f} graph")
        del keys, rows
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
