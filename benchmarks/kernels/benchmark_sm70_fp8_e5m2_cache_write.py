# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Benchmark the native E5M2 reshape-and-cache writer on SM70."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch


def _measure(run, warmup: int, reps: int, inner_reps: int) -> dict[str, float]:
    for _ in range(warmup):
        run()
    torch.accelerator.synchronize()

    samples = []
    for _ in range(reps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        run()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / inner_reps)
    return {
        "median_us": statistics.median(samples),
        "mean_us": statistics.mean(samples),
        "min_us": min(samples),
        "max_us": max(samples),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--num-tokens", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=1)
    parser.add_argument("--head-size", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--num-blocks", type=int, default=1024)
    parser.add_argument("--layout", choices=("NHD", "HND"), default="NHD")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--reps", type=int, default=1000)
    parser.add_argument("--inner-reps", type=int, default=1)
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    torch.ops.load_library(str(args.library.resolve()))
    device = torch.device("cuda")
    key = torch.randn(
        args.num_tokens,
        args.num_heads,
        args.head_size,
        device=device,
        dtype=torch.float16,
    )
    value = torch.randn_like(key)
    logical_shape = (
        args.num_blocks,
        args.block_size,
        args.num_heads,
        args.head_size,
    )
    if args.layout == "NHD":
        key_cache = torch.zeros(logical_shape, device=device, dtype=torch.uint8)
        value_cache = torch.zeros_like(key_cache)
    else:
        physical_shape = (
            args.num_blocks,
            args.num_heads,
            args.block_size,
            args.head_size,
        )
        key_cache = torch.zeros(
            physical_shape, device=device, dtype=torch.uint8
        ).permute(0, 2, 1, 3)
        value_cache = torch.zeros(
            physical_shape, device=device, dtype=torch.uint8
        ).permute(0, 2, 1, 3)
    slot_mapping = torch.arange(args.num_tokens, device=device, dtype=torch.int64)
    scale = torch.full((1,), args.scale, device=device, dtype=torch.float32)

    def launch() -> None:
        torch.ops._C_cache_ops.reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            "fp8_e5m2",
            scale,
            scale,
        )

    def run_many() -> None:
        for _ in range(args.inner_reps):
            launch()

    if args.cuda_graph:
        for _ in range(3):
            run_many()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run_many()
        run = graph.replay
    else:
        run = run_many

    result = {
        "library": str(args.library.resolve()),
        "device": torch.cuda.get_device_name(),
        "num_tokens": args.num_tokens,
        "num_heads": args.num_heads,
        "head_size": args.head_size,
        "block_size": args.block_size,
        "layout": args.layout,
        "scale": args.scale,
        "cuda_graph": args.cuda_graph,
        "inner_reps": args.inner_reps,
        **_measure(run, args.warmup, args.reps, args.inner_reps),
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
