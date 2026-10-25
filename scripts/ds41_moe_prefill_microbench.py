#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One stage-0 MoE layer's EXL3 row-tiled prefill pass on one GPU, with the LIVE routing, for ncu / nsys
(scripts/moe_prefill_profile.sh) and as a plain microbench. No checkpoint: the trellis words are random (the decode
is data-independent), the per-expert K (bits) come from the live franken4 plan and the expert -> rank assignment and
token routing from the v1-ab24 routing histograms, so the kernel sees exactly the live tile mix of TP rank 0.

  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD python scripts/ds41_moe_prefill_microbench.py [--layer 5] [--T 1016]
      [--iters 10] [--dense-min 160] [--ksplit 1] [--layout 1] [--tiers 16,32] [--once]
--once runs a single pass (for ncu --launch-count); otherwise prints ms per pass and the tile / dense counts."""

from __future__ import annotations

import os
import argparse
import json
import sys
from types import SimpleNamespace

import numpy as np
import torch

from vllm.models.deepseek_v41.common.contracts import HIDDEN, MOE_INTER, N_EXPERTS, TOP_K
from vllm.models.deepseek_v41.sm70 import moe_exl3 as mx

PLAN = os.environ.get("DS41_EXL3_PLAN", "exl3_moe_plan.json")
HIST = os.environ.get("VLLM_DS41_MOE_EXL3_BALANCE", "v100/balance/v1-ab24")


def build(layer_id: int, tp_rank: int, tp_size: int, dev, args):
    cfg = json.load(open(PLAN))["layers"][str(layer_id)]
    bpw = np.array([x[1] for x in cfg], dtype=np.int64)
    hist = np.load(f"{HIST}/layer{layer_id:02d}.npy").astype(np.float64)
    counts = hist[1] if hist.ndim == 2 else hist
    owner = mx.assign_experts(bpw / bpw.sum(), counts, tp_size)
    local = np.flatnonzero(owner == tp_rank)
    kl = bpw[local]
    H, inter = HIDDEN, MOE_INTER
    w13 = np.array([mx._words(H, inter, int(k)) for k in kl], dtype=np.int64)
    w2 = np.array([mx._words(inter, H, int(k)) for k in kl], dtype=np.int64)
    offs13 = np.concatenate([[0], np.cumsum(w13)[:-1]])
    offs2 = np.concatenate([[0], np.cumsum(w2)[:-1]])
    loc_of = np.full(N_EXPERTS, -1, dtype=np.int64)
    loc_of[local] = np.arange(len(local))
    g = torch.Generator(device=dev).manual_seed(layer_id)
    n_loc = len(local)
    layer = SimpleNamespace(
        exl3_H=H, exl3_inter=inter, exl3_kbits_host=kl, exl3_offs13_host=offs13, exl3_offs2_host=offs2,
        exl3_kbits=torch.as_tensor(kl, dtype=torch.int32, device=dev),
        exl3_offs13=torch.as_tensor(offs13, dtype=torch.int64, device=dev),
        exl3_offs2=torch.as_tensor(offs2, dtype=torch.int64, device=dev),
        exl3_glob2loc=torch.as_tensor(loc_of, dtype=torch.int32, device=dev),
        w13_trellis=torch.randint(-32768, 32767, (2, int(w13.sum()) * 2), dtype=torch.int16, device=dev, generator=g),
        w2_trellis=torch.randint(-32768, 32767, (int(w2.sum()) * 2,), dtype=torch.int16, device=dev, generator=g),
        w13_suh=(torch.randn(2, n_loc, H, device=dev, generator=g) * 0.05).half(),
        w13_svh=(torch.randn(2, n_loc, inter, device=dev, generator=g) * 0.05).half(),
        w2_suh=(torch.randn(n_loc, inter, device=dev, generator=g) * 0.05).half(),
        w2_svh=(torch.randn(n_loc, H, device=dev, generator=g) * 0.05).half(),
        exl3_limit=10.0)
    method = mx.DeepseekV41Exl3MoEMethod.__new__(mx.DeepseekV41Exl3MoEMethod)
    method.__dict__.update(prefill_dense_min=args.dense_min, tiled_layout=args.layout, decode_ksplit=4,
                           prefill_ksplit=args.ksplit, prefill_tiers=tuple(int(v) for v in args.tiers.split(",")),
                           layer_id=layer_id, tp_rank=tp_rank, tp_size=tp_size)
    # token routing drawn from the layer's histogram: T tokens x top-6 distinct experts each
    p = counts / counts.sum()
    rng = np.random.default_rng(layer_id)
    ids = np.stack([rng.choice(N_EXPERTS, size=TOP_K, replace=False, p=p) for _ in range(args.T)]).astype(np.int32)
    topk_ids = torch.as_tensor(ids, device=dev)
    x = (torch.randn(args.T, H, device=dev, generator=g) * 0.5).half()
    nbytes = sum(t.numel() * t.element_size() for t in (layer.w13_trellis, layer.w2_trellis))
    rows_here = int(np.isin(ids, local).sum())
    return method, layer, x, topk_ids, nbytes, rows_here, n_loc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=5)
    ap.add_argument("--T", type=int, default=1016)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--dense-min", type=int, default=160)
    ap.add_argument("--ksplit", type=int, default=1)
    ap.add_argument("--layout", type=int, default=1)
    ap.add_argument("--tiers", default="16,32")
    ap.add_argument("--tp-rank", type=int, default=0)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    dev = torch.device("cuda")
    method, layer, x, topk_ids, nbytes, rows_here, n_loc = build(args.layer, args.tp_rank, 4, dev, args)
    run = lambda: method._expert_slots_tiled(layer, x, topk_ids)
    if args.once:
        run()
        torch.cuda.synchronize()
        return
    run(); torch.cuda.synchronize()
    a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(args.iters):
        run()
    b.record(); torch.cuda.synchronize()
    ms = a.elapsed_time(b) / args.iters
    weights = float(n_loc) * 3 * HIDDEN * MOE_INTER            # gate + up + down per expert
    print(f"layer {args.layer} tp_rank {args.tp_rank}: {n_loc} local experts, {rows_here} of {args.T * TOP_K} routed rows "
          f"land here, trellis {nbytes / 2**30:.2f} GiB (~{weights / 1e9:.2f} G weights)")
    print(f"  tiled pass: {ms:.2f} ms  |  HBM floor {nbytes / 900e9 * 1e3:.2f} ms  |  decode-ALU floor "
          f"{weights / 2.5e12 * 1e3:.2f} ms (2.5e12 weights/s)  |  HMMA floor "
          f"{2 * rows_here * 3 * HIDDEN * MOE_INTER / 100e12 * 1e3:.2f} ms (100 TFLOPS)")


if __name__ == "__main__":
    sys.exit(main())
