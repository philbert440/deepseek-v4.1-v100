# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare GEMM+all-reduce with pipelined GEMM+RS+AG on SM70.

Launch this benchmark with torchrun.  It models the exact Qwen3.8 TP4
row-parallel down/output projection boundary without loading the model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed._symmetric_memory import enable_symm_mem_for_group


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=8000)
    parser.add_argument("--k", type=int, default=4352)
    parser.add_argument("--n", type=int, default=5120)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--trials", type=int, default=7)
    parser.add_argument("--chunk-counts", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--chunk-axis", choices=("m", "n"), default="m")
    parser.add_argument("--skip-symm-mem", action="store_true")
    parser.add_argument("--include-gemma-norm", action="store_true")
    parser.add_argument("--norm-epsilon", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--json-out", type=Path, required=True)
    return parser.parse_args()


def _event_trials(
    launch: Callable[[], None], warmup: int, iters: int, trials: int
) -> dict[str, Any]:
    for _ in range(warmup):
        launch()
    torch.accelerator.synchronize()

    samples: list[float] = []
    for _ in range(trials):
        dist.barrier()
        start = torch.Event(enable_timing=True)
        end = torch.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            launch()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end) / iters))
    return {
        "median_ms": float(statistics.median(samples)),
        "mean_ms": float(statistics.mean(samples)),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": samples,
    }


def _digest(tensor: torch.Tensor) -> str:
    raw = tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def main() -> int:
    args = _parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.accelerator.set_device_index(local_rank)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 4:
        raise RuntimeError("This benchmark requires TP4.")
    if torch.cuda.get_device_capability(local_rank) != (7, 0):
        raise RuntimeError("This benchmark requires SM70/V100 GPUs.")
    if args.m % world_size:
        raise ValueError("M must be divisible by the world size.")
    chunk_extent = args.m if args.chunk_axis == "m" else args.n
    if any(chunks <= 0 or chunk_extent % chunks for chunks in args.chunk_counts):
        raise ValueError(
            f"Every chunk count must be positive and divide {args.chunk_axis.upper()}."
        )
    if args.include_gemma_norm:
        import vllm.kernels  # noqa: F401

        if args.n != 5120:
            raise ValueError("The SM70 Gemma long-prefill norm requires N=5120.")
        if not hasattr(torch.ops._C, "sm70_gemma_long_prefill_fused_add_rms_norm"):
            raise RuntimeError("The active extension lacks the SM70 Gemma norm op.")

    group_name = dist.group.WORLD.group_name
    if not args.skip_symm_mem:
        enable_symm_mem_for_group(group_name)
    device = torch.device("cuda", local_rank)
    generator = torch.Generator(device=device).manual_seed(args.seed + rank)
    inputs = torch.randn(
        (args.m, args.k),
        generator=generator,
        device=device,
        dtype=torch.float16,
    ).mul_(0.1)
    weight = torch.randn(
        (args.k, args.n),
        generator=generator,
        device=device,
        dtype=torch.float16,
    ).mul_(0.01)
    baseline_out = torch.empty((args.m, args.n), device=device, dtype=torch.float16)
    pipelined_out = torch.empty_like(baseline_out)
    plain_sp_mm_out = torch.empty_like(baseline_out)
    plain_sp_out = torch.empty_like(baseline_out)
    plain_sp_shard = torch.empty(
        (args.m // world_size, args.n), device=device, dtype=torch.float16
    )
    chunked_outputs = {
        chunks: torch.empty_like(baseline_out) for chunks in args.chunk_counts
    }
    chunked_n_comm_outputs: dict[int, list[torch.Tensor]] = {}
    if args.chunk_axis == "n":
        chunked_n_comm_outputs = {
            chunks: [
                torch.empty(
                    (args.m, args.n // chunks),
                    device=device,
                    dtype=torch.float16,
                )
                for _ in range(chunks)
            ]
            for chunks in args.chunk_counts
        }
    residual = None
    norm_weight = None
    baseline_normalized = None
    baseline_residual_out = None
    pipelined_normalized_shard = None
    pipelined_residual_shard = None
    chunked_normalized: dict[int, torch.Tensor] = {}
    chunked_residual_out: dict[int, torch.Tensor] = {}
    if args.include_gemma_norm:
        residual = torch.randn(
            (args.m, args.n),
            generator=torch.Generator(device=device).manual_seed(args.seed + 10000),
            device=device,
            dtype=torch.float32,
        ).mul_(0.1)
        norm_weight = torch.randn(
            (args.n,),
            generator=torch.Generator(device=device).manual_seed(args.seed + 20000),
            device=device,
            dtype=torch.bfloat16,
        ).mul_(0.01)
        baseline_normalized = torch.empty_like(baseline_out)
        baseline_residual_out = torch.empty_like(residual)
        pipelined_normalized_shard = torch.empty(
            (args.m // world_size, args.n), device=device, dtype=torch.float16
        )
        pipelined_residual_shard = torch.empty(
            (args.m // world_size, args.n), device=device, dtype=torch.float32
        )
        chunked_normalized = {
            chunks: torch.empty_like(baseline_out) for chunks in args.chunk_counts
        }
        chunked_residual_out = {
            chunks: torch.empty_like(residual) for chunks in args.chunk_counts
        }

    def norm_out(
        output: torch.Tensor,
        residual_output: torch.Tensor,
        input: torch.Tensor,
        input_residual: torch.Tensor,
    ) -> None:
        assert norm_weight is not None
        torch.ops._C.sm70_gemma_long_prefill_fused_add_rms_norm(
            output,
            residual_output,
            input,
            input_residual,
            norm_weight,
            args.norm_epsilon,
        )

    def baseline() -> None:
        torch.mm(inputs, weight, out=baseline_out)
        dist.all_reduce(baseline_out)
        if args.include_gemma_norm:
            assert baseline_normalized is not None
            assert baseline_residual_out is not None
            assert residual is not None
            norm_out(
                baseline_normalized,
                baseline_residual_out,
                baseline_out,
                residual,
            )

    def pipelined() -> None:
        shard = torch.ops.symm_mem.fused_matmul_reduce_scatter(
            inputs,
            weight,
            "sum",
            0,
            group_name,
        )
        if args.include_gemma_norm:
            assert residual is not None
            assert pipelined_normalized_shard is not None
            assert pipelined_residual_shard is not None
            residual_shard = residual.narrow(
                0, rank * (args.m // world_size), args.m // world_size
            )
            norm_out(
                pipelined_normalized_shard,
                pipelined_residual_shard,
                shard,
                residual_shard,
            )
            shard = pipelined_normalized_shard
        dist.all_gather_into_tensor(pipelined_out, shard)

    def plain_sp() -> None:
        torch.mm(inputs, weight, out=plain_sp_mm_out)
        dist.reduce_scatter_tensor(plain_sp_shard, plain_sp_mm_out)
        shard = plain_sp_shard
        if args.include_gemma_norm:
            assert residual is not None
            assert pipelined_normalized_shard is not None
            assert pipelined_residual_shard is not None
            residual_shard = residual.narrow(
                0, rank * (args.m // world_size), args.m // world_size
            )
            norm_out(
                pipelined_normalized_shard,
                pipelined_residual_shard,
                shard,
                residual_shard,
            )
            shard = pipelined_normalized_shard
        dist.all_gather_into_tensor(plain_sp_out, shard)

    def make_chunked_all_reduce(chunks: int) -> Callable[[], None]:
        if args.chunk_axis == "m":
            lhs_chunks = inputs.chunk(chunks, dim=0)
            rhs_chunks = (weight,) * chunks
            output_chunks = chunked_outputs[chunks].chunk(chunks, dim=0)
        else:
            lhs_chunks = (inputs,) * chunks
            rhs_chunks = weight.chunk(chunks, dim=1)
            output_chunks = chunked_n_comm_outputs[chunks]
        result_chunks = chunked_outputs[chunks].chunk(
            chunks, dim=0 if args.chunk_axis == "m" else 1
        )

        def launch() -> None:
            works = []
            for lhs, rhs, output_chunk in zip(lhs_chunks, rhs_chunks, output_chunks):
                torch.mm(lhs, rhs, out=output_chunk)
                works.append(dist.all_reduce(output_chunk, async_op=True))
            for work, output_chunk, result_chunk in zip(
                works, output_chunks, result_chunks
            ):
                work.wait()
                if args.chunk_axis == "n":
                    result_chunk.copy_(output_chunk)
            if args.include_gemma_norm:
                assert residual is not None
                norm_out(
                    chunked_normalized[chunks],
                    chunked_residual_out[chunks],
                    chunked_outputs[chunks],
                    residual,
                )

        return launch

    baseline_timing = _event_trials(baseline, args.warmup, args.iters, args.trials)
    plain_sp_timing = _event_trials(plain_sp, args.warmup, args.iters, args.trials)
    chunked_launches = {
        chunks: make_chunked_all_reduce(chunks) for chunks in args.chunk_counts
    }
    chunked_timings = {
        chunks: _event_trials(launch, args.warmup, args.iters, args.trials)
        for chunks, launch in chunked_launches.items()
    }
    pipelined_timing = None
    if not args.skip_symm_mem:
        pipelined_timing = _event_trials(
            pipelined, args.warmup, args.iters, args.trials
        )
    baseline()
    plain_sp()
    for launch in chunked_launches.values():
        launch()
    if not args.skip_symm_mem:
        pipelined()
    torch.accelerator.synchronize()

    if args.include_gemma_norm:
        assert baseline_normalized is not None
        baseline_result = baseline_normalized
    else:
        baseline_result = baseline_out
    local_payload = {
        "rank": rank,
        "baseline": baseline_timing,
        "baseline_sha256": _digest(baseline_result),
        "plain_sp": None,
        "pipelined": None,
        "chunked": {},
    }
    plain_sp_difference = (baseline_result.float() - plain_sp_out.float()).abs()
    local_payload["plain_sp"] = {
        "timing": plain_sp_timing,
        "speedup": baseline_timing["median_ms"] / plain_sp_timing["median_ms"],
        "max_abs_diff": float(plain_sp_difference.max().item()),
        "mean_abs_diff": float(plain_sp_difference.mean().item()),
        "equal_elements": int((baseline_result == plain_sp_out).sum().item()),
        "elements": baseline_result.numel(),
        "sha256": _digest(plain_sp_out),
    }
    if pipelined_timing is not None:
        difference = (baseline_result.float() - pipelined_out.float()).abs()
        local_payload["pipelined"] = {
            "timing": pipelined_timing,
            "speedup": baseline_timing["median_ms"] / pipelined_timing["median_ms"],
            "max_abs_diff": float(difference.max().item()),
            "mean_abs_diff": float(difference.mean().item()),
            "equal_elements": int((baseline_result == pipelined_out).sum().item()),
            "elements": baseline_result.numel(),
            "sha256": _digest(pipelined_out),
        }
    for chunks, output in chunked_outputs.items():
        chunked_result = (
            chunked_normalized[chunks] if args.include_gemma_norm else output
        )
        chunked_difference = (baseline_result.float() - chunked_result.float()).abs()
        local_payload["chunked"][str(chunks)] = {
            "timing": chunked_timings[chunks],
            "speedup": baseline_timing["median_ms"]
            / chunked_timings[chunks]["median_ms"],
            "max_abs_diff": float(chunked_difference.max().item()),
            "mean_abs_diff": float(chunked_difference.mean().item()),
            "equal_elements": int((baseline_result == chunked_result).sum().item()),
            "elements": baseline_result.numel(),
            "sha256": _digest(chunked_result),
        }
    gathered: list[dict[str, Any] | None] = [None] * world_size
    dist.all_gather_object(gathered, local_payload)
    if rank == 0:
        payload = {
            "environment": {
                "world_size": world_size,
                "device": torch.cuda.get_device_name(local_rank),
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "m": args.m,
                "k": args.k,
                "n": args.n,
                "dtype": "float16",
                "warmup": args.warmup,
                "iters": args.iters,
                "trials": args.trials,
                "chunk_counts": args.chunk_counts,
                "chunk_axis": args.chunk_axis,
                "symm_mem": not args.skip_symm_mem,
                "include_gemma_norm": args.include_gemma_norm,
                "norm_epsilon": args.norm_epsilon,
                "seed": args.seed,
                "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
                "nccl_algo": os.getenv("NCCL_ALGO"),
                "nccl_proto": os.getenv("NCCL_PROTO"),
                "nccl_min_nchannels": os.getenv("NCCL_MIN_NCHANNELS"),
                "nccl_max_nchannels": os.getenv("NCCL_MAX_NCHANNELS"),
            },
            "ranks": gathered,
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps(payload, indent=2))
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
