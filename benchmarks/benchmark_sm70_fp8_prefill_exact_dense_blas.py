# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark the exact-dense SM70 FP8 prefill path with a selected BLAS backend.

This loads the real TP-local Qwen3.8 gate-up and down-projection weights.  Run
each BLAS backend in a fresh process so global ATen and cuBLAS caches cannot
contaminate the comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import statistics
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

import vllm


def _load_vllm_c_extension() -> None:
    if "vllm._C" in sys.modules:
        vllm._C = sys.modules["vllm._C"]
        return
    candidate = os.getenv("VLLM_BENCH_C_EXTENSION")
    if candidate is None:
        importlib.import_module("vllm._C")
        return
    spec = importlib.util.spec_from_file_location("vllm._C", candidate)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load candidate extension: {candidate}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["vllm._C"] = module
    vllm._C = module
    spec.loader.exec_module(module)


_load_vllm_c_extension()

from vllm import _sm70_ops as sm70_ops  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
    )
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--tp-rank", type=int, default=0)
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=("gate_up", "down", "attention_output"),
        default=("gate_up", "down"),
    )
    parser.add_argument("--m", type=int, default=8000)
    parser.add_argument(
        "--backend",
        choices=("exact-dense", "resident-turbomind-f16"),
        default="exact-dense",
    )
    parser.add_argument(
        "--gated-silu",
        action="store_true",
        help="Benchmark gate_up with the fused interleaved SiLU-and-mul output.",
    )
    parser.add_argument(
        "--blas-library",
        choices=("default", "cublas", "cublaslt"),
        required=True,
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--trials", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--cuda-graph", action="store_true")
    return parser.parse_args()


def _load_keys(model: Path, keys: list[str]) -> dict[str, torch.Tensor]:
    index = json.loads((model / "model.safetensors.index.json").read_text())
    weight_map: dict[str, str] = index["weight_map"]
    by_file: dict[str, list[str]] = {}
    for key in keys:
        by_file.setdefault(weight_map[key], []).append(key)

    result: dict[str, torch.Tensor] = {}
    for filename, file_keys in by_file.items():
        with safe_open(model / filename, framework="pt", device="cpu") as handle:
            for key in file_keys:
                result[key] = handle.get_tensor(key)
    return result


def _column_shard(
    weight: torch.Tensor,
    scales: torch.Tensor,
    tp_size: int,
    tp_rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    n = int(weight.shape[0])
    if n % tp_size:
        raise ValueError(f"N={n} is not divisible by TP={tp_size}.")
    begin = tp_rank * (n // tp_size)
    end = begin + n // tp_size
    return (
        weight.view(torch.uint8)[begin:end].contiguous(),
        scales[begin // 128 : math.ceil(end / 128)].contiguous(),
    )


def _row_shard(
    weight: torch.Tensor,
    scales: torch.Tensor,
    tp_size: int,
    tp_rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    k = int(weight.shape[1])
    if k % tp_size:
        raise ValueError(f"K={k} is not divisible by TP={tp_size}.")
    begin = tp_rank * (k // tp_size)
    end = begin + k // tp_size
    return (
        weight.view(torch.uint8)[:, begin:end].contiguous(),
        scales[:, begin // 128 : math.ceil(end / 128)].contiguous(),
    )


def _load_case(
    model: Path,
    case: str,
    tp_size: int,
    tp_rank: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    if case == "attention_output":
        prefixes = ["model.language_model.layers.3.self_attn.o_proj"]
    else:
        root = "model.language_model.layers.1.mlp"
        projections = ("gate_proj", "up_proj") if case == "gate_up" else ("down_proj",)
        prefixes = [f"{root}.{projection}" for projection in projections]
    keys = [
        key
        for prefix in prefixes
        for key in (f"{prefix}.weight", f"{prefix}.weight_scale_inv")
    ]
    loaded = _load_keys(model, keys)

    raw_parts: list[torch.Tensor] = []
    scale_parts: list[torch.Tensor] = []
    fp8_dtype: torch.dtype | None = None
    for prefix in prefixes:
        weight = loaded[f"{prefix}.weight"]
        scales = loaded[f"{prefix}.weight_scale_inv"].float()
        fp8_dtype = weight.dtype if fp8_dtype is None else fp8_dtype
        if weight.dtype != fp8_dtype:
            raise ValueError("Fused gate-up projections must use one FP8 dtype.")
        shard = (
            _column_shard(weight, scales, tp_size, tp_rank)
            if case == "gate_up"
            else _row_shard(weight, scales, tp_size, tp_rank)
        )
        raw_parts.append(shard[0])
        scale_parts.append(shard[1])

    assert fp8_dtype is not None
    raw = torch.cat(raw_parts, dim=0)
    scales = torch.cat(scale_parts, dim=0)
    return (
        raw.to(device).view(fp8_dtype).contiguous(),
        scales.to(device).contiguous(),
        prefixes,
    )


def _event_trials(
    launch: Callable[[], None], warmup: int, iters: int, trials: int
) -> dict[str, Any]:
    for _ in range(warmup):
        launch()
    torch.accelerator.synchronize()
    samples: list[float] = []
    for _ in range(trials):
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


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_case(
    args: argparse.Namespace, case: str, device: torch.device
) -> dict[str, Any]:
    qweight, scales, prefixes = _load_case(
        args.model, case, args.tp_size, args.tp_rank, device
    )
    n, k = (int(dim) for dim in qweight.shape)
    expected = {
        "gate_up": (8704, 5120),
        "down": (5120, 4352),
        "attention_output": (5120, 1536),
    }[case]
    if (n, k) != expected:
        raise ValueError(f"Unexpected {case} TP-local shape N={n}, K={k}.")

    tm_weight, tm_scales, meta = sm70_ops.fp8_sm70_prepare(qweight, scales, 128)
    k_ld, q_ld = (int(value) for value in meta.tolist())
    generator = torch.Generator(device=device).manual_seed(args.seed + n + k)
    inputs = torch.randn(
        (args.m, k), generator=generator, device=device, dtype=torch.float16
    ).mul_(0.1)
    output = torch.empty((args.m, n), device=device, dtype=torch.float16)
    if args.gated_silu and case == "gate_up":
        output = torch.empty((args.m, n // 2), device=device, dtype=torch.float16)
    resident_bytes = 0
    if args.backend == "resident-turbomind-f16":
        dense_weight = torch.empty((k, n), device=device, dtype=torch.float16)
        sm70_ops.fp8_sm70_dequantize_out(dense_weight, tm_weight, tm_scales, 128)
        source_weight = dense_weight.t().contiguous()
        f16_tm_weight, f16_meta = sm70_ops.sm70_f16_prepare(source_weight)
        f16_k_ld = int(f16_meta[0].item())
        resident_bytes = f16_tm_weight.numel() * f16_tm_weight.element_size()

        def eager_launch() -> None:
            sm70_ops.sm70_f16_gemm_out(
                output,
                inputs,
                f16_tm_weight,
                f16_k_ld,
                args.gated_silu and case == "gate_up",
            )

    else:
        workspace = torch.empty((k, n), device=device, dtype=torch.float16)
        resident_bytes = workspace.numel() * workspace.element_size()

        def eager_launch() -> None:
            sm70_ops.fp8_gemm_sm70_prefill_dispatch_out(
                output,
                workspace.data_ptr(),
                inputs,
                tm_weight,
                tm_scales,
                128,
                k_ld,
                q_ld,
                args.gated_silu and case == "gate_up",
                3920,
            )

    cutlass_env = "VLLM_SM70_FP8_PREFILL_CUTLASS"
    selected_cutlass = os.environ.get(cutlass_env)
    os.environ[cutlass_env] = "0"
    eager_launch()
    torch.accelerator.synchronize(device)
    reference = output.clone()
    if selected_cutlass is None:
        os.environ.pop(cutlass_env)
    else:
        os.environ[cutlass_env] = selected_cutlass

    launch = eager_launch
    graph: torch.cuda.CUDAGraph | None = None
    if args.cuda_graph:
        for _ in range(3):
            eager_launch()
        torch.accelerator.synchronize(device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            eager_launch()
        launch = graph.replay

    timing = _event_trials(launch, args.warmup, args.iters, args.trials)
    launch()
    torch.accelerator.synchronize(device)
    output_float = output.float()
    reference_float = reference.float()
    return {
        "case": case,
        "prefixes": prefixes,
        "m": args.m,
        "n": n,
        "k": k,
        "timing": timing,
        "useful_tflops": 2 * args.m * n * k / (timing["median_ms"] * 1e-3) / 1e12,
        "output_sha256": _digest(output),
        "reference_sha256": _digest(reference),
        "output_bitwise_equal": bool(torch.equal(output, reference)),
        "output_max_abs_diff": float(
            torch.max(torch.abs(output_float - reference_float)).item()
        ),
        "output_finite": bool(torch.isfinite(output).all().item()),
        "resident_bytes": resident_bytes,
    }


def main() -> int:
    args = _parse_args()
    device = torch.device(args.device)
    if torch.cuda.get_device_capability(device) != (7, 0):
        raise RuntimeError("This benchmark requires an SM70/V100 GPU.")
    torch.accelerator.set_device_index(device.index or 0)
    if args.blas_library != "default":
        torch.backends.cuda.preferred_blas_library(args.blas_library)

    results = [_run_case(args, case, device) for case in args.cases]
    extension_path = Path(vllm._C.__file__).resolve()
    payload = {
        "environment": {
            "model": str(args.model),
            "device": torch.cuda.get_device_name(device),
            "capability": list(torch.cuda.get_device_capability(device)),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "blas_library_requested": args.blas_library,
            "blas_library_effective": str(torch.backends.cuda.preferred_blas_library()),
            "gated_silu": args.gated_silu,
            "backend": args.backend,
            "cuda_graph": args.cuda_graph,
            "cutlass": os.getenv("VLLM_SM70_FP8_PREFILL_CUTLASS", "<unset:default-on>"),
            "vllm_c_extension": str(extension_path),
            "vllm_c_extension_sha256": _file_digest(extension_path),
            "tp_size": args.tp_size,
            "tp_rank": args.tp_rank,
            "seed": args.seed,
        },
        "results": results,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
