# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    marlin_make_workspace_new,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    rand_marlin_weight_mxfp4_like,
)
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types

pytestmark = pytest.mark.skipif(
    not current_platform.is_device_capability(70),
    reason="The FP16 MXFP4 kernels under test are specific to SM70.",
)

_M = 1
_N = 512
_K = 1024
_GROUP_SIZE = 32


def _make_fixed_e8m0_case(
    monkeypatch: pytest.MonkeyPatch, exponent: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(20260816)
    original_randint = torch.randint

    def fixed_randint(low, high, size, **kwargs):
        if low == 110 and high == 120 and kwargs.get("dtype") == torch.uint8:
            return torch.full(
                size,
                exponent,
                dtype=torch.uint8,
                device=kwargs.get("device"),
            )
        return original_randint(low, high, size, **kwargs)

    shape_only_weight = torch.empty((_N, _K), dtype=torch.float16, device="cuda")
    with monkeypatch.context() as context:
        context.setattr(torch, "randint", fixed_randint)
        weight_ref, qweight, scales = rand_marlin_weight_mxfp4_like(
            shape_only_weight, _GROUP_SIZE
        )

    activation = torch.ones((_M, _K), dtype=torch.float16, device="cuda")
    reference = torch.matmul(activation.float(), weight_ref.float()).half()
    return activation, qweight, scales, reference


def _assert_matches_reference(output: torch.Tensor, reference: torch.Tensor) -> None:
    if torch.count_nonzero(reference).item() == 0:
        assert torch.count_nonzero(output).item() == 0
        return
    torch.testing.assert_close(output, reference, rtol=0, atol=0)


@pytest.mark.parametrize("exponent", [102, 103, 112, 113])
@pytest.mark.parametrize("split_k", [1, 8])
def test_sm70_dense_mxfp4_preserves_e8m0_subnormals(
    monkeypatch: pytest.MonkeyPatch, exponent: int, split_k: int
) -> None:
    activation, qweight, scales, reference = _make_fixed_e8m0_case(
        monkeypatch, exponent
    )
    workspace = marlin_make_workspace_new(activation.device)
    output = torch.empty((_M, _N), dtype=torch.float16, device="cuda")
    monkeypatch.setenv("SM70_MARLIN_DENSE_SPLIT_K", str(split_k))

    result = ops.marlin_gemm(
        activation,
        output,
        qweight,
        None,
        scales,
        None,
        None,
        None,
        None,
        None,
        workspace,
        scalar_types.float4_e2m1f,
        _M,
        _N,
        _K,
        is_k_full=True,
        use_atomic_add=False,
        use_fp32_reduce=True,
        is_zp_float=False,
    )
    _assert_matches_reference(result, reference)


@pytest.mark.parametrize("exponent", [102, 103, 112, 113])
@pytest.mark.parametrize("split_k", [1, 8])
def test_sm70_moe_mxfp4_preserves_e8m0_subnormals(
    monkeypatch: pytest.MonkeyPatch, exponent: int, split_k: int
) -> None:
    activation, qweight, scales, reference = _make_fixed_e8m0_case(
        monkeypatch, exponent
    )
    topk_ids = torch.zeros((_M, 1), dtype=torch.int64, device="cuda")
    sorted_ids, expert_ids, padded = moe_align_block_size(topk_ids, 8, 1)
    topk_weights = torch.ones((_M, 1), dtype=torch.float32, device="cuda")
    workspace = marlin_make_workspace_new(activation.device, 4)
    output = torch.empty((_M, _N), dtype=torch.float16, device="cuda")
    monkeypatch.setenv("SM70_MARLIN_MOE_SPLIT_K", str(split_k))

    result = ops.moe_wna16_marlin_gemm(
        activation,
        output,
        qweight.unsqueeze(0),
        None,
        scales.unsqueeze(0),
        None,
        None,
        None,
        None,
        None,
        workspace,
        sorted_ids,
        expert_ids,
        padded,
        topk_weights,
        8,
        1,
        False,
        scalar_types.float4_e2m1f,
        _M,
        _N,
        _K,
        True,
        False,
        True,
        False,
    )
    _assert_matches_reference(result, reference)
