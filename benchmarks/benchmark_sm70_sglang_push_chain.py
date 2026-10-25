# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark SGLang's push all-reduce on a verifier-shaped graph chain.

Run this script from a pinned SGLang checkout and environment. It deliberately
uses the public SGLang JIT communicator so that the comparison includes its
production graph-input registration and two-epoch push protocol.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--numel", type=int, default=8 * 5120)
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=40)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--max-push-blocks", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-out", type=Path, required=True)
    return parser.parse_args()


def _max_rank_float(value: float) -> float:
    result = torch.tensor([value], dtype=torch.float64)
    dist.all_reduce(result, op=dist.ReduceOp.MAX)
    return float(result.item())


def main() -> None:
    args = _parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 4:
        raise ValueError(f"this verifier microbenchmark requires TP4, got {world_size}")

    torch.accelerator.set_device_index(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="gloo")

    import sglang.srt.distributed.parallel_state as parallel_state
    from sglang.jit_kernel.all_reduce import (
        AllReduceAlgo,
        _jit_custom_all_reduce_push_module,
    )
    from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
        CustomAllReduceV2,
    )

    parallel_state._WORLD = parallel_state.init_world_group(
        ranks=list(range(world_size)),
        local_rank=local_rank,
        backend="nccl",
    )
    cpu_group = parallel_state._WORLD.cpu_group
    nccl_group = parallel_state._WORLD.device_group
    if nccl_group is None:
        raise RuntimeError("SGLang did not create the NCCL device group")

    bytes_per_input = args.numel * torch.float16.itemsize
    communicator = CustomAllReduceV2(
        cpu_group,
        device,
        max_pull_size=bytes_per_input,
        max_push_size=bytes_per_input,
        max_push_blocks=args.max_push_blocks,
    )
    if communicator.disabled:
        raise RuntimeError("SGLang push all-reduce is disabled on this topology")
    communicator.override_algo = AllReduceAlgo.ONE_SHOT_PUSH
    _jit_custom_all_reduce_push_module(torch.float16, world_size)

    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + rank)
    source = (
        torch.rand(
            (args.count, args.numel),
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        * 0.03
        + 0.01
    ).half()
    work = source.clone()

    gathered = [torch.empty_like(source) for _ in range(world_size)]
    dist.all_gather(gathered, source, group=nccl_group)
    reference = gathered[0].float()
    for peer in gathered[1:]:
        reference.add_(peer.float())
    reference = reference.half()

    graph = torch.cuda.CUDAGraph()
    with communicator.capture(), torch.cuda.graph(graph):
        for index in range(args.count):
            communicator.custom_all_reduce(work[index])
    torch.accelerator.synchronize()
    dist.barrier()

    work.copy_(source)
    graph.replay()
    torch.accelerator.synchronize()
    exact = bool(torch.equal(work, reference))
    difference = (work.float() - reference.float()).abs()
    mismatch_count = int((work != reference).sum().item())

    for _ in range(args.warmup):
        graph.replay()
    torch.accelerator.synchronize()
    dist.barrier()

    start = torch.Event(enable_timing=True)
    end = torch.Event(enable_timing=True)
    start.record()
    for _ in range(args.iters):
        graph.replay()
    end.record()
    end.synchronize()
    local_ms = start.elapsed_time(end) / args.iters
    max_rank_ms = _max_rank_float(local_ms)

    if rank == 0:
        result = {
            "implementation": "sglang_v100_one_shot_push",
            "world_size": world_size,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "numel": args.numel,
            "bytes_per_input": bytes_per_input,
            "allreduces_per_graph_replay": args.count,
            "max_push_blocks": args.max_push_blocks,
            "warmup_replays": args.warmup,
            "timed_replays": args.iters,
            "max_rank_ms_per_graph_replay": max_rank_ms,
            "max_rank_us_per_allreduce": max_rank_ms * 1000.0 / args.count,
            "exact_rank_order_fp32_reference": exact,
            "mismatch_count": mismatch_count,
            "max_diff": float(difference.max().item()),
            "mean_diff": float(difference.mean().item()),
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))

    communicator.close()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
