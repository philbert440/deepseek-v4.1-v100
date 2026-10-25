// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#pragma once

#include <torch/all.h>

namespace vllm::awq_sm70 {

bool sm70_fp8_prefill_cutlass_out(torch::Tensor out, torch::Tensor in_feats,
                                  torch::Tensor dense_weight, bool gated_silu);

}  // namespace vllm::awq_sm70
