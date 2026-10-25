# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One-GPU microbench of the prefill sparse attention per layer call: today's tiled path (GATHER_ONCE 0 / 1) vs the
fused kernel (VLLM_DS41_ATTN_FUSED_PREFILL, P split on / off). Same process, same inputs.

Shapes of one TP4 rank on one prefill chunk: T queries, 16 local heads, 128 window rows (consecutive positions through
a shuffled 64-row block table) + 512 compressed rows per query out of ctx / ratio visible, D = 512. ``--ckv random``
draws every query's top-512 independently (worst case for L2 reuse); ``shared`` gives runs of 16 consecutive queries
the same set (closer to real top-k overlap). Reports ms/layer, peak transient MiB, and the error of every path vs an
FP32 torch oracle on the first --check queries (rel RMS and fp16 ulps of max(|ref|, row RMS)).

    CUDA_VISIBLE_DEVICES=0 python scripts/ds41_fused_prefill_bench.py --T 1024 --ctx 32768 --ratio 2
    CUDA_VISIBLE_DEVICES=0 python scripts/ds41_fused_prefill_bench.py --T 1024 --ctx 131072 --ratio 2 --ckv shared
"""

from __future__ import annotations

import argparse

import torch

from vllm.models.deepseek_v41.common import qat
from vllm.models.deepseek_v41.sm70 import sparse_kernels as sk


def build(T: int, H: int, ctx: int, ratio: int, ckv_mode: str, window_only: bool, dev: str = "cuda"):
    g = torch.Generator(device=dev).manual_seed(0)
    D, bs = 512, 64
    n_win = ((ctx + T + bs - 1) // bs) * bs
    win = (torch.randn(n_win, D, device=dev, generator=g) * 0.5).half()
    perm = torch.randperm(n_win // bs, device=dev, generator=g)
    pos = ctx + torch.arange(T, device=dev)                                    # query positions in the chunk
    wp = pos[:, None] - 127 + torch.arange(128, device=dev)[None, :]
    wi = perm[wp // bs] * bs + wp % bs                                         # window rows through the block table
    sources = [(win, wi)]
    if not window_only:
        n_ckv = max(512, ctx // ratio)
        ckv = qat.fp4_e4m3_pack_torch(torch.randn(n_ckv, D, device=dev, generator=g) * 2)
        n_sets = T if ckv_mode == "random" else (T + 15) // 16
        sets = torch.rand(n_sets, n_ckv, device=dev, generator=g).topk(512, dim=1).indices.sort(dim=1).values
        ci = sets if ckv_mode == "random" else sets.repeat_interleave(16, dim=0)[:T]
        sources.append((ckv, ci.contiguous()))
    q = (torch.randn(T, H, D, device=dev, generator=g) * 0.05).half()
    sink = torch.randn(H, device=dev, generator=g)
    return q, sources, sink


def err(out: torch.Tensor, ref: torch.Tensor) -> tuple[float, float]:
    d = (out.float() - ref.float()).abs()
    rms = ref.float().pow(2).mean(dim=-1, keepdim=True).sqrt()
    unit = torch.exp2(torch.floor(torch.log2(torch.maximum(ref.float().abs(), rms).clamp(min=2.0 ** -14))) - 10)
    return (d.pow(2).mean().sqrt() / ref.float().pow(2).mean().sqrt()).item(), (d / unit).max().item()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=1024)
    ap.add_argument("--ctx", type=int, default=32768, help="tokens before the chunk")
    ap.add_argument("--ratio", type=int, default=2, help="compression ratio (stage 0 layers 2-19: 2; 20-39: 1)")
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--ckv", choices=("random", "shared"), default="random")
    ap.add_argument("--window-only", action="store_true", help="SWA-only layer (0, 1): no compressed source")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--check", type=int, default=256, help="queries scored against the FP32 oracle")
    a = ap.parse_args()
    q, sources, sink = build(a.T, a.heads, a.ctx, a.ratio, a.ckv, a.window_only)
    D = 512
    nc = 0 if a.window_only else 512
    print(f"T={a.T} H={a.heads} ctx={a.ctx} ratio={a.ratio} ckv={a.ckv} keys/query={128 + nc}")
    variants = {"tiled GATHER_ONCE=0": dict(fused=False, once=False),
                "tiled GATHER_ONCE=1": dict(fused=False, once=True),
                "fused PSPLIT=1": dict(fused=True, psplit=True),
                "fused PSPLIT=0": dict(fused=True, psplit=False)}
    ref = torch.empty_like(q[: a.check])
    sk.sparse_attention(q[: a.check], [(r, i[: a.check]) for r, i in sources], sink, D ** -0.5, ref, impl="torch")
    res = {}
    for name, v in variants.items():
        sk.FUSED_PREFILL = v["fused"]
        sk.GATHER_ONCE = v.get("once", False)
        sk.FUSED_PREFILL_PSPLIT = v.get("psplit", True)
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
        ms = e0.elapsed_time(e1) / a.iters
        peak = (torch.cuda.max_memory_allocated() - base) / 2 ** 20
        rr, ulp = err(out[: a.check], ref)
        res[name] = (ms, peak, rr, ulp)
    base_ms, base_rr = res["tiled GATHER_ONCE=0"][0], res["tiled GATHER_ONCE=0"][2]
    for name, (ms, peak, rr, ulp) in res.items():
        print(f"{name:22s} {ms:8.3f} ms/layer  x{base_ms / ms:5.1f}  peak {peak:7.1f} MiB  "
              f"rel_rms {rr:.2e} ({rr / base_rr:4.2f}x today)  max {ulp:5.2f} ulp")
    flops = 4 * a.T * a.heads * (128 + nc) * D
    ms = res["fused PSPLIT=1"][0]
    print(f"fused PSPLIT=1: {flops / ms / 1e9:.1f} TFLOP/s useful; per 20-layer stage at this shape: "
          f"{20 * ms:.1f} ms vs today {20 * base_ms:.1f} ms")


if __name__ == "__main__":
    main()
