# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

import vllm.model_executor.layers.quantization.awq as awq_module
import vllm.model_executor.layers.quantization.awq_triton as awq_triton
from vllm.model_executor.layers.quantization.awq import AWQConfig, AWQLinearMethod


class _FakePlatform:
    def __init__(self, *, is_cuda: bool, capability: int | None):
        self._is_cuda = is_cuda
        self._capability = capability

    def is_cuda(self) -> bool:
        return self._is_cuda

    def is_device_capability(self, capability: int) -> bool:
        return self._capability == capability


@pytest.mark.parametrize(
    ("platform", "expected_route"),
    [
        (_FakePlatform(is_cuda=True, capability=70), "triton"),
        (_FakePlatform(is_cuda=True, capability=60), "classic"),
        (_FakePlatform(is_cuda=True, capability=75), "classic"),
        (_FakePlatform(is_cuda=False, capability=70), "classic"),
        (_FakePlatform(is_cuda=False, capability=None), "classic"),
    ],
)
def test_awq_fallback_uses_triton_only_on_sm70(
    monkeypatch: pytest.MonkeyPatch,
    platform: _FakePlatform,
    expected_route: str,
) -> None:
    calls: list[str] = []
    weight = torch.arange(16, dtype=torch.float32).reshape(2, 8)

    def triton_dequantize(*_args):
        calls.append("triton")
        return weight

    def classic_gemm(x, *_args):
        calls.append("classic")
        return torch.matmul(x, weight)

    monkeypatch.setattr(awq_module, "current_platform", platform)
    monkeypatch.setattr(awq_triton, "awq_dequantize_triton", triton_dequantize)
    monkeypatch.setattr(awq_module.ops, "awq_gemm", classic_gemm)

    method = AWQLinearMethod(AWQConfig(4, 128, True))
    layer = SimpleNamespace(
        qweight=torch.empty((2, 1), dtype=torch.int32),
        scales=torch.empty((1, 8), dtype=torch.float32),
        qzeros=torch.empty((1, 1), dtype=torch.int32),
    )
    x = torch.ones((1, 2), dtype=torch.float32)

    output = method.apply(layer, x)

    assert calls == [expected_route]
    torch.testing.assert_close(output, torch.matmul(x, weight))
