# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Microbenchmark DFlash2 target GDN metadata construction on SM70."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import torch

from vllm import envs
from vllm.triton_utils import triton
from vllm.v1.attention.backends.gdn_attn import (
    DFlash2GDNGroupDescriptor,
    _dflash2_gdn_group_metadata_kernel,
    _get_ddtree_gdn_fast_common_buffers,
    build_gdn_spec_decode_state_contract,
    prepare_dflash2_gdn_group_metadata,
)
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.worker.gpu.attn_utils import compute_common_gdn_attn_metadata


@dataclass
class _MicroBuilder:
    num_spec_state_tokens: int
    decode_cudagraph_max_bs: int
    spec_state_indices_tensor: torch.Tensor
    _ddtree_fast_common_buffers: Any
    use_full_cuda_graph: bool = True
    vllm_config: Any = None

    def __post_init__(self) -> None:
        self.vllm_config = SimpleNamespace(
            cache_config=SimpleNamespace(mamba_cache_mode="none")
        )


def _percentile(samples: list[float], quantile: float) -> float:
    ordered = sorted(samples)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def _distribution(samples: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(samples),
        "p50": _percentile(samples, 0.50),
        "p90": _percentile(samples, 0.90),
        "p99": _percentile(samples, 0.99),
        "min": min(samples),
        "max": max(samples),
    }


def _profile_cuda_launches(step: Any) -> int:
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        step()
        torch.accelerator.synchronize()
    return sum(
        1
        for event in prof.events()
        if str(getattr(event, "device_type", "")).lower().endswith("cuda")
    )


def _measure(step: Any, repeats: int) -> dict[str, dict[str, float]]:
    wall_samples: list[float] = []
    gpu_samples: list[float] = []
    start = torch.Event(enable_timing=True)
    end = torch.Event(enable_timing=True)
    for _ in range(repeats):
        torch.accelerator.synchronize()
        wall_start = time.perf_counter_ns()
        start.record()
        step()
        end.record()
        end.synchronize()
        wall_samples.append((time.perf_counter_ns() - wall_start) / 1e6)
        gpu_samples.append(start.elapsed_time(end))
    return {
        "synchronized_wall_ms": _distribution(wall_samples),
        "cuda_event_ms": _distribution(gpu_samples),
    }


def _run_case(
    batch_size: int,
    num_groups: int,
    width: int,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    device = torch.device("cuda")
    num_actual_tokens = batch_size * width
    common_buffers = _get_ddtree_gdn_fast_common_buffers(
        device,
        num_actual_tokens,
        width,
    )
    builders = [
        _MicroBuilder(
            num_spec_state_tokens=width - 1,
            decode_cudagraph_max_bs=num_actual_tokens,
            spec_state_indices_tensor=torch.empty(
                (num_actual_tokens, width), dtype=torch.int32, device=device
            ),
            _ddtree_fast_common_buffers=common_buffers,
        )
        for _ in range(num_groups)
    ]
    block_tables = tuple(
        (
            torch.arange(
                batch_size * width,
                dtype=torch.int32,
                device=device,
            ).view(batch_size, width)
            + group_id * 10_000
        ).contiguous()
        for group_id in range(num_groups)
    )
    query_start_loc = torch.arange(
        0,
        num_actual_tokens + 1,
        width,
        dtype=torch.int32,
        device=device,
    )
    query_start_loc_cpu = query_start_loc.cpu()
    draft_counts_cpu = torch.full((batch_size,), width - 1, dtype=torch.int32)
    common_metadata = compute_common_gdn_attn_metadata(
        num_decode_draft_tokens_cpu=draft_counts_cpu,
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc_cpu,
        num_spec_state_tokens=width - 1,
        legacy_mixed_decode_routing=False,
    )
    assert common_metadata is not None
    accepted_values = [3, 5, 7, 2]
    accepted = torch.tensor(
        [accepted_values[index % len(accepted_values)] for index in range(batch_size)],
        dtype=torch.int32,
        device=device,
    )
    seq_lens = torch.full((batch_size,), 4097, dtype=torch.int32, device=device)
    spec_mask_cpu = torch.ones(batch_size, dtype=torch.bool)
    spec_mask = common_metadata.spec_sequence_masks

    def legacy_step() -> None:
        for group_id, builder in enumerate(builders):
            contract = build_gdn_spec_decode_state_contract(
                block_table_tensor=block_tables[group_id],
                seq_lens=seq_lens,
                block_size=16,
                num_spec=width - 1,
                spec_sequence_masks_cpu=spec_mask_cpu,
                num_accepted_tokens=accepted,
                current_state_block_ids=None,
                is_mamba_cache_all=False,
            )
            builder.spec_state_indices_tensor[:batch_size].copy_(
                contract.spec_state_indices_tensor,
                non_blocking=True,
            )
            builder.spec_state_indices_tensor[batch_size:].fill_(PAD_SLOT_ID)

            common_buffers.spec_sequence_masks[:batch_size].copy_(
                spec_mask, non_blocking=True
            )
            common_buffers.spec_sequence_masks[batch_size:].fill_(False)
            common_buffers.spec_token_indx[:num_actual_tokens].copy_(
                common_metadata.spec_token_indx,
                non_blocking=True,
            )
            common_buffers.spec_query_start_loc[: batch_size + 1].copy_(
                common_metadata.spec_query_start_loc,
                non_blocking=True,
            )
            spec_num_query_tokens = common_metadata.spec_query_start_loc[-1]
            common_buffers.spec_query_start_loc[batch_size + 1 :].fill_(
                spec_num_query_tokens
            )
            common_buffers.num_accepted_tokens[:batch_size].copy_(
                contract.num_accepted_tokens,
                non_blocking=True,
            )
            common_buffers.num_accepted_tokens[batch_size:].fill_(1)
            common_buffers.spec_state_slot_selectors[:batch_size].copy_(
                contract.spec_state_slot_selectors,
                non_blocking=True,
            )
            common_buffers.spec_state_slot_selectors[batch_size:].fill_(1)

    descriptor: DFlash2GDNGroupDescriptor | None = None

    def fused_step() -> None:
        nonlocal descriptor
        result = prepare_dflash2_gdn_group_metadata(
            builders_by_group=[
                (group_id, builder) for group_id, builder in enumerate(builders)
            ],
            block_tables=block_tables,
            common_gdn_metadata=common_metadata,
            num_accepted_tokens=accepted,
            num_actual_tokens=num_actual_tokens,
            descriptor=descriptor,
        )
        if result is None:
            raise RuntimeError("fused DFlash2 GDN metadata path was not eligible")
        _, descriptor = result

    def fused_kernel_only_step() -> None:
        if descriptor is None:
            raise RuntimeError("fused descriptor has not been initialized")
        block = triton.next_power_of_2(
            max(num_actual_tokens * width, num_actual_tokens + 1)
        )
        _dflash2_gdn_group_metadata_kernel[(num_groups,)](
            descriptor.block_table_ptrs,
            descriptor.state_output_ptrs,
            descriptor.block_table_strides,
            common_metadata.spec_query_start_loc,
            accepted,
            accepted,
            common_buffers.spec_sequence_masks,
            common_buffers.spec_query_start_loc,
            common_buffers.num_accepted_tokens,
            common_buffers.spec_state_slot_selectors,
            batch_size,
            num_actual_tokens,
            WIDTH=width,
            PAD_ID=PAD_SLOT_ID,
            BLOCK=block,
            num_warps=1,
        )

    def fused_with_host_fence_step() -> None:
        fused_step()
        if (
            common_metadata.spec_query_start_loc[-1].item()
            != common_metadata.num_spec_decode_tokens
        ):
            raise AssertionError("invalid speculative query span")

    legacy_step()
    torch.accelerator.synchronize()
    expected_states = [
        builder.spec_state_indices_tensor.clone() for builder in builders
    ]
    expected_common = (
        common_buffers.spec_sequence_masks.clone(),
        common_buffers.spec_query_start_loc.clone(),
        common_buffers.num_accepted_tokens.clone(),
        common_buffers.spec_state_slot_selectors.clone(),
    )
    for builder in builders:
        builder.spec_state_indices_tensor.fill_(777)
    common_buffers.spec_sequence_masks.fill_(True)
    common_buffers.spec_query_start_loc.fill_(-1)
    common_buffers.num_accepted_tokens.fill_(-1)
    common_buffers.spec_state_slot_selectors.fill_(-1)
    fused_step()
    torch.accelerator.synchronize()
    for actual_builder, expected_state in zip(builders, expected_states):
        torch.testing.assert_close(
            actual_builder.spec_state_indices_tensor,
            expected_state,
            rtol=0,
            atol=0,
        )
    for actual, expected in zip(
        (
            common_buffers.spec_sequence_masks,
            common_buffers.spec_query_start_loc,
            common_buffers.num_accepted_tokens,
            common_buffers.spec_state_slot_selectors,
        ),
        expected_common,
    ):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    for _ in range(warmup):
        legacy_step()
        fused_step()
    torch.accelerator.synchronize()

    legacy_launches = _profile_cuda_launches(legacy_step)
    fused_launches = _profile_cuda_launches(fused_step)
    fused_with_host_fence_launches = _profile_cuda_launches(fused_with_host_fence_step)
    fused_kernel_only_launches = _profile_cuda_launches(fused_kernel_only_step)
    legacy = _measure(legacy_step, repeats)
    fused = _measure(fused_step, repeats)
    fused_with_host_fence = _measure(fused_with_host_fence_step, repeats)
    fused_kernel_only = _measure(fused_kernel_only_step, repeats)
    return {
        "batch_size": batch_size,
        "num_actual_tokens": num_actual_tokens,
        "num_groups": num_groups,
        "state_width": width,
        "correctness": "bitwise_equal_after_poison",
        "legacy_cuda_launches": legacy_launches,
        "fused_cuda_launches": fused_launches,
        "fused_with_host_fence_cuda_launches": fused_with_host_fence_launches,
        "fused_kernel_only_cuda_launches": fused_kernel_only_launches,
        "legacy": legacy,
        "fused": fused,
        "fused_with_host_fence": fused_with_host_fence,
        "fused_kernel_only": fused_kernel_only,
        "host_fence_wall_cost_p50_ms": (
            fused_with_host_fence["synchronized_wall_ms"]["p50"]
            - fused["synchronized_wall_ms"]["p50"]
        ),
        "fused_python_overhead_p50_ms": (
            fused["synchronized_wall_ms"]["p50"]
            - fused_kernel_only["synchronized_wall_ms"]["p50"]
        ),
        "wall_speedup": (
            legacy["synchronized_wall_ms"]["p50"] / fused["synchronized_wall_ms"]["p50"]
        ),
        "gpu_speedup": (legacy["cuda_event_ms"]["p50"] / fused["cuda_event_ms"]["p50"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--num-groups", type=int, default=10)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if torch.cuda.get_device_capability() != (7, 0):
        raise RuntimeError("this microbenchmark requires SM70")
    os.environ["VLLM_SM70_DFLASH2_FUSED_GDN_METADATA"] = "1"
    os.environ["VLLM_SM70_DFLASH2_GDN_METADATA_SHADOW"] = "0"
    envs.disable_envs_cache()

    results = [
        _run_case(
            batch_size=batch_size,
            num_groups=args.num_groups,
            width=args.width,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        for batch_size in args.batch_sizes
    ]
    print(
        json.dumps(
            {"device": torch.cuda.get_device_name(), "cases": results},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
