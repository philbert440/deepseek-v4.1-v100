# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Benchmark the DFlash2 draft sliding-window paged-attention shape."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch
from flash_attn_v100 import flash_attn_prefill_paged


def _time_ms(fn, *, warmup: int, reps: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples = []
    for _ in range(reps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seq-lens",
        type=int,
        nargs="+",
        default=[1024, 32768, 131072, 261888],
    )
    parser.add_argument("--q-len", type=int, default=8)
    parser.add_argument("--num-query-heads", type=int, default=8)
    parser.add_argument("--num-kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--window-left", type=int, default=2048)
    parser.add_argument("--window-right", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--reps", type=int, default=100)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tensor-output", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if torch.cuda.get_device_capability() != (7, 0):
        raise RuntimeError("This benchmark targets SM70")

    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.float16
    max_seq_len = max(args.seq_lens)
    num_blocks = (max_seq_len + args.block_size - 1) // args.block_size
    query = torch.randn(
        1,
        args.q_len,
        args.num_query_heads,
        args.head_dim,
        device=device,
        dtype=dtype,
    )
    key_cache = torch.randn(
        num_blocks,
        args.block_size,
        args.num_kv_heads,
        args.head_dim,
        device=device,
        dtype=dtype,
    )
    value_cache = torch.randn_like(key_cache)
    full_block_table = torch.arange(
        num_blocks, device=device, dtype=torch.int32
    ).unsqueeze(0)
    output = torch.empty_like(query)
    tensor_outputs: dict[int, torch.Tensor] = {}
    results = []

    for seq_len in args.seq_lens:
        active_blocks = (seq_len + args.block_size - 1) // args.block_size
        block_table = full_block_table[:, :active_blocks]
        seq_lens = torch.tensor([seq_len], device=device, dtype=torch.int32)

        def run(
            block_table: torch.Tensor = block_table,
            seq_lens: torch.Tensor = seq_lens,
        ) -> None:
            flash_attn_prefill_paged(
                query,
                key_cache,
                value_cache,
                block_table,
                seq_lens,
                out=output,
                causal=False,
                window_size=(args.window_left, args.window_right),
            )

        samples = _time_ms(run, warmup=args.warmup, reps=args.reps)
        run()
        torch.cuda.synchronize()
        tensor_outputs[seq_len] = output.detach().cpu().clone()
        results.append(
            {
                "seq_len": seq_len,
                "median_ms": statistics.median(samples),
                "mean_ms": statistics.mean(samples),
                "min_ms": min(samples),
                "max_ms": max(samples),
                "five_draft_layers_median_ms": 5 * statistics.median(samples),
            }
        )

    report = {
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "shape": {
            "q_len": args.q_len,
            "num_query_heads": args.num_query_heads,
            "num_kv_heads": args.num_kv_heads,
            "head_dim": args.head_dim,
            "block_size": args.block_size,
            "causal": False,
            "window_size": [args.window_left, args.window_right],
        },
        "warmup": args.warmup,
        "reps": args.reps,
        "results": results,
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if args.tensor_output is not None:
        args.tensor_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(tensor_outputs, args.tensor_output)


if __name__ == "__main__":
    main()
