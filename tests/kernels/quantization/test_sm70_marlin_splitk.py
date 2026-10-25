# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    marlin_make_workspace_new,
    marlin_permute_scales,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    gptq_pack,
    gptq_quantize_weights,
)
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types

pytestmark = pytest.mark.skipif(
    not current_platform.is_device_capability(70),
    reason="The split-K FP32 reduction is specific to SM70.",
)


def test_sm70_marlin_splitk_fp32_reduction(monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(20260816)
    size_m, size_n, size_k = 1, 512, 1024
    group_size = 128
    quant_type = scalar_types.uint4b8
    device = torch.device("cuda")

    activation = torch.randn((size_m, size_k), dtype=torch.float16, device=device)
    weight = torch.randn(
        (size_k, size_n), dtype=torch.float16, device=device
    ) / math.sqrt(size_k)
    weight_ref, qweight, scales, _, _ = gptq_quantize_weights(
        weight, quant_type, group_size, False
    )
    packed_weight = gptq_pack(qweight, quant_type.size_bits, size_k, size_n)
    marlin_weight = ops.gptq_marlin_repack(
        packed_weight,
        torch.empty(0, dtype=torch.int32, device=device),
        size_k,
        size_n,
        quant_type.size_bits,
        False,
    )
    marlin_scales = marlin_permute_scales(scales, size_k, size_n, group_size)
    workspace = marlin_make_workspace_new(device)
    output = torch.empty((size_m, size_n), dtype=torch.float16, device=device)

    monkeypatch.setenv("SM70_MARLIN_DENSE_CTA_GEOMETRY", "32x128x32x4x32x32x32")
    monkeypatch.setenv("SM70_MARLIN_DENSE_METADATA_CACHE", "vector_words")

    def run(split_k: int) -> torch.Tensor:
        monkeypatch.setenv("SM70_MARLIN_DENSE_SPLIT_K", str(split_k))
        return ops.marlin_gemm(
            activation,
            output,
            marlin_weight,
            None,
            marlin_scales,
            None,
            None,
            None,
            None,
            None,
            workspace,
            quant_type,
            size_m,
            size_n,
            size_k,
            is_k_full=True,
            use_atomic_add=False,
            use_fp32_reduce=True,
            is_zp_float=False,
        ).clone()

    reference = torch.matmul(activation.float(), weight_ref.float())
    split1 = run(1)
    split8_outputs = [run(8) for _ in range(20)]
    torch.cuda.synchronize()

    split1_rms = torch.sqrt(torch.mean((split1.float() - reference).square()))
    split8_rms = torch.sqrt(
        torch.mean((split8_outputs[0].float() - reference).square())
    )
    max_eager_spread = max(
        (item.float() - split8_outputs[0].float()).abs().max().item()
        for item in split8_outputs
    )
    assert torch.isfinite(split8_outputs[0]).all()
    assert split8_rms <= split1_rms * 1.05
    assert max_eager_spread <= 0.001953125

    graph = torch.cuda.CUDAGraph()
    run(8)
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        ops.marlin_gemm(
            activation,
            output,
            marlin_weight,
            None,
            marlin_scales,
            None,
            None,
            None,
            None,
            None,
            workspace,
            quant_type,
            size_m,
            size_n,
            size_k,
            is_k_full=True,
            use_atomic_add=False,
            use_fp32_reduce=True,
            is_zp_float=False,
        )
    graph_outputs = []
    for _ in range(20):
        graph.replay()
        graph_outputs.append(output.clone())
    torch.cuda.synchronize()
    max_graph_spread = max(
        (item.float() - graph_outputs[0].float()).abs().max().item()
        for item in graph_outputs
    )
    assert max_graph_spread <= 0.001953125
