# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One-GPU microbench of the prefill sparse attention, VLLM_DS41_ATTN_GATHER_ONCE off vs on (same process).

Shapes of one TP4 rank on one 1024-token chunk: 16 local heads, 128 window rows + 512 compressed rows per query,
tile 256 (VLLM_DS41_ATTN_TILE). Reports ms per layer call, per-layer peak allocation above the inputs, and bitwise
equality of the outputs. (The per-group row-id sharing is not in this bench: it saves n-1 ``logical_to_rows``
calls per group of n layers.)

    CUDA_VISIBLE_DEVICES=0 python scripts/ds41_gather_once_bench.py [--T 1024] [--ctx 131072] [--iters 20]
"""

from __future__ import annotations

import argparse

import torch

from vllm.models.deepseek_v41.common import qat
from vllm.models.deepseek_v41.sm70 import sparse_kernels as sk


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=1024)
    ap.add_argument("--ctx", type=int, default=131072, help="compressed rows visible (ratio-2 layers: ctx/2)")
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--iters", type=int, default=20)
    a = ap.parse_args()
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(0)
    T, H, D = a.T, a.heads, 512
    n_win, n_ckv = 1 << 16, a.ctx
    win = (torch.randn(n_win, D, device=dev, generator=g) * 0.5).half()
    ckv = qat.fp4_e4m3_pack_torch(torch.randn(n_ckv, D, device=dev, generator=g) * 2)
    wi = torch.randint(0, n_win, (T, 128), device=dev, generator=g)
    ci = torch.sort(torch.randint(0, n_ckv, (T, 512), device=dev, generator=g), dim=1).values
    ci[: T // 8, 400:] = -1                                   # early queries see fewer than 512 entries
    q = (torch.randn(T, H, D, device=dev, generator=g) * 0.05).half()
    sink = torch.randn(H, device=dev, generator=g)
    sources = [(win, wi), (ckv, ci)]
    outs, res = {}, {}
    for on in (False, True):
        sk.GATHER_ONCE = on
        out = torch.empty_like(q)
        for _ in range(3):
            sk.sparse_attention(q, sources, sink, D ** -0.5, out, impl="sm70", tile=a.tile)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(a.iters):
            sk.sparse_attention(q, sources, sink, D ** -0.5, out, impl="sm70", tile=a.tile)
        e1.record()
        torch.cuda.synchronize()
        res[on] = (e0.elapsed_time(e1) / a.iters, (torch.cuda.max_memory_allocated() - base) / 2 ** 20)
        outs[on] = out
    same = torch.equal(outs[False].view(torch.int16), outs[True].view(torch.int16))
    for on in (False, True):
        print(f"GATHER_ONCE={int(on)}: {res[on][0]:7.3f} ms/layer  peak transient {res[on][1]:7.1f} MiB")
    print(f"saving {res[False][0] - res[True][0]:.3f} ms/layer/chunk; outputs bit-identical: {same}")
    raise SystemExit(0 if same else 1)


if __name__ == "__main__":
    main()
