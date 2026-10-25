# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark SM70 fused reduce-scatter + Gemma RMSNorm + all-gather.

Launch with torchrun on a fully connected TP4 V100 group.  The candidate is a
benchmark-only implementation of the communication topology in NVIDIA's NCCL
Device API fused RMSNorm example, using vLLM's existing IPC buffers and
cross-device signal substrate so it can be measured with NCCL 2.27.
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

import vllm.kernels  # noqa: F401
from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=8000)
    parser.add_argument("--n", type=int, default=5120)
    parser.add_argument("--threads", type=int, nargs="+", default=[256, 512, 1024])
    parser.add_argument("--blocks", type=int, nargs="+", default=[80])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--candidate-cuda-graph", action="store_true")
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
    cpu_group = dist.new_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 4:
        raise RuntimeError("This benchmark requires TP4.")
    if torch.cuda.get_device_capability(local_rank) != (7, 0):
        raise RuntimeError("This benchmark requires SM70/V100 GPUs.")
    if args.n != 5120 or args.m % world_size:
        raise ValueError("The candidate requires N=5120 and TP-divisible M.")
    if any(threads not in (256, 512, 1024) for threads in args.threads):
        raise ValueError("--threads choices must be 256, 512, or 1024.")
    if any(blocks < 1 or blocks > 512 for blocks in args.blocks):
        raise ValueError("--blocks choices must be in [1, 512].")
    variants = [(threads, blocks) for threads in args.threads for blocks in args.blocks]

    device = torch.device("cuda", local_rank)
    generator = torch.Generator(device=device).manual_seed(args.seed + rank)
    partial = torch.randn(
        (args.m, args.n),
        generator=generator,
        device=device,
        dtype=torch.float16,
    ).mul_(0.1)
    residual = torch.randn(
        (args.m, args.n),
        generator=torch.Generator(device=device).manual_seed(args.seed + 10000),
        device=device,
        dtype=torch.float32,
    ).mul_(0.1)
    weight = torch.randn(
        (args.n,),
        generator=torch.Generator(device=device).manual_seed(args.seed + 20000),
        device=device,
        dtype=torch.float16,
    ).mul_(0.01)

    baseline_reduced = torch.empty_like(partial)
    baseline_normalized = torch.empty_like(partial)
    baseline_residual = torch.empty_like(residual)
    candidate_normalized = {variant: torch.empty_like(partial) for variant in variants}
    candidate_residual = {
        variant: torch.empty_like(residual, dtype=torch.float32) for variant in variants
    }

    max_size = partial.numel() * partial.element_size() + 1
    pynccl = PyNcclCommunicator(group=cpu_group, device=device)
    if pynccl.disabled:
        raise RuntimeError("PyNccl is unavailable for the production baseline.")
    communicator = CustomAllreduce(
        cpu_group,
        device,
        max_size=max_size,
        long_prefill_fusion_enabled=True,
    )
    if communicator.disabled:
        raise RuntimeError("Custom all-reduce is unavailable for this TP4 group.")

    def norm_out(
        output: torch.Tensor,
        residual_output: torch.Tensor,
        input: torch.Tensor,
        input_residual: torch.Tensor,
    ) -> None:
        torch.ops._C.sm70_gemma_long_prefill_fused_add_rms_norm(
            output,
            residual_output,
            input,
            input_residual,
            weight,
            args.epsilon,
        )

    def baseline() -> None:
        pynccl.all_reduce(partial, baseline_reduced)
        norm_out(
            baseline_normalized,
            baseline_residual,
            baseline_reduced,
            residual,
        )

    def baseline_all_reduce() -> None:
        pynccl.all_reduce(partial, baseline_reduced)

    def baseline_norm() -> None:
        norm_out(
            baseline_normalized,
            baseline_residual,
            baseline_reduced,
            residual,
        )

    def d2d_copy() -> None:
        baseline_reduced.copy_(partial)

    def make_candidate(threads: int, blocks: int) -> Callable[[], None]:
        def launch() -> None:
            os.environ["VLLM_SM70_TP4_LONG_FUSED_NORM_THREADS"] = str(threads)
            os.environ["VLLM_SM70_TP4_LONG_FUSED_NORM_BLOCKS"] = str(blocks)
            communicator.sm70_tp4_reduce_scatter_gemma_rms_norm_all_gather(
                partial,
                residual,
                weight,
                args.epsilon,
                normalized_out=candidate_normalized[(threads, blocks)],
                residual_out=candidate_residual[(threads, blocks)],
            )

        return launch

    baseline_timing = _event_trials(baseline, args.warmup, args.iters, args.trials)
    baseline_ar_timing = _event_trials(
        baseline_all_reduce, args.warmup, args.iters, args.trials
    )
    baseline_all_reduce()
    baseline_norm_timing = _event_trials(
        baseline_norm, args.warmup, args.iters, args.trials
    )
    copy_timing = _event_trials(d2d_copy, args.warmup, args.iters, args.trials)
    candidate_launches = {variant: make_candidate(*variant) for variant in variants}
    candidate_graphs: list[torch.cuda.CUDAGraph] = []
    if args.candidate_cuda_graph:
        graph_launches: dict[tuple[int, int], Callable[[], None]] = {}
        for variant, launch in candidate_launches.items():
            launch()
            torch.accelerator.synchronize()
            dist.barrier()
            graph = torch.cuda.CUDAGraph()
            with communicator.capture(), torch.cuda.graph(graph):
                launch()
            candidate_graphs.append(graph)
            graph_launches[variant] = graph.replay
        candidate_launches = graph_launches
    candidate_timings = {
        variant: _event_trials(launch, args.warmup, args.iters, args.trials)
        for variant, launch in candidate_launches.items()
    }

    baseline()
    for launch in candidate_launches.values():
        launch()
    torch.accelerator.synchronize()
    local_rows = args.m // world_size
    residual_reference = baseline_residual.narrow(0, rank * local_rows, local_rows)
    candidates: dict[str, Any] = {}
    for variant in variants:
        threads, blocks = variant
        normalized_difference = (
            baseline_normalized.float() - candidate_normalized[variant].float()
        ).abs()
        candidate_residual_shard = candidate_residual[variant].narrow(
            0, rank * local_rows, local_rows
        )
        residual_difference = (residual_reference - candidate_residual_shard).abs()
        timing = candidate_timings[variant]
        candidates[f"t{threads}_b{blocks}"] = {
            "threads": threads,
            "blocks": blocks,
            "timing": timing,
            "speedup": baseline_timing["median_ms"] / timing["median_ms"],
            "normalized_max_abs_diff": float(normalized_difference.max().item()),
            "normalized_mean_abs_diff": float(normalized_difference.mean().item()),
            "normalized_equal_elements": int(
                (baseline_normalized == candidate_normalized[variant]).sum().item()
            ),
            "normalized_elements": baseline_normalized.numel(),
            "residual_max_abs_diff": float(residual_difference.max().item()),
            "residual_mean_abs_diff": float(residual_difference.mean().item()),
            "normalized_sha256": _digest(candidate_normalized[variant]),
            "residual_shard_sha256": _digest(candidate_residual_shard),
        }

    local_payload = {
        "rank": rank,
        "baseline": baseline_timing,
        "baseline_all_reduce": baseline_ar_timing,
        "baseline_norm": baseline_norm_timing,
        "d2d_copy": copy_timing,
        "baseline_normalized_sha256": _digest(baseline_normalized),
        "baseline_residual_shard_sha256": _digest(residual_reference),
        "candidates": candidates,
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
                "n": args.n,
                "dtype": "float16",
                "residual_dtype": "float32",
                "threads": args.threads,
                "blocks": args.blocks,
                "candidate_cuda_graph": args.candidate_cuda_graph,
                "warmup": args.warmup,
                "iters": args.iters,
                "trials": args.trials,
                "epsilon": args.epsilon,
                "seed": args.seed,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "nccl_algo": os.environ.get("NCCL_ALGO"),
                "nccl_proto": os.environ.get("NCCL_PROTO"),
            },
            "ranks": gathered,
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps(payload, indent=2), flush=True)

    communicator.close()
    pynccl.destroy()
    dist.barrier()
    dist.destroy_process_group(cpu_group)
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
