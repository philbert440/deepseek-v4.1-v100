// SPDX-License-Identifier: Apache-2.0
// Exact sparse Philox sampling for the SM70 Qwen3.8 batch-one top-20 route.

#include <torch/all.h>

#include <ATen/core/TransformationHelper.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/cuda/Exceptions.h>
#include <ATen/cuda/PhiloxUtils.cuh>
#include <c10/cuda/CUDAGuard.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <cstdint>
#include <mutex>
#include <optional>

namespace {

constexpr int kTopK = 20;
constexpr int kWarpSize = 32;
constexpr int kNativeBlockSize = 256;
constexpr int kNativeUnroll = 4;
constexpr int kMaxGeneratorOffsetsPerCurandCall = 4;

__device__ __forceinline__ bool top1_better(float value, int token,
                                            float best_value, int best_token) {
  return value > best_value || (value == best_value && token < best_token);
}

__device__ __forceinline__ float native_exponential_for_index(
    int64_t index, int64_t total_threads, at::PhiloxCudaState philox_args) {
  auto [seed, offset] = at::cuda::philox::unpack(philox_args);
  const int64_t values_per_round = total_threads * kNativeUnroll;
  const int64_t round = index / values_per_round;
  const int64_t within_round = index - round * values_per_round;
  const uint64_t subsequence =
      static_cast<uint64_t>(within_round % total_threads);
  const int component = static_cast<int>(within_round / total_threads);

  curandStatePhilox4_32_10_t state;
  curand_init(seed, subsequence, offset, &state);
  float4 random = make_float4(0.f, 0.f, 0.f, 0.f);
  for (int64_t iteration = 0; iteration <= round; ++iteration) {
    random = curand_uniform4(&state);
  }
  const float uniform = reinterpret_cast<const float*>(&random)[component];
  return at::transformation::exponential<float>(uniform, 1.f);
}

template <bool EmitMetadata, bool ChunkedIndices>
__global__ void sm70_sample_sorted_top20_philox_kernel(
    int64_t* sampled_token_out, int64_t* sparse_ids_out,
    float* sparse_probs_out, const float* top_values,
    const int64_t* top_indices, const int64_t* local_indices,
    const int64_t* global_positions, int64_t local_candidate_count,
    int chunk_size, int64_t vocab_size, int64_t total_threads, float top_p,
    at::PhiloxCudaState philox_args) {
  const int lane = threadIdx.x;
  __shared__ float selected_values[kTopK];
  __shared__ int64_t selected_ids[kTopK];
  __shared__ float selected_random[kTopK];

  int64_t lane_token64 = -1;
  if (lane < kTopK) {
    if (ChunkedIndices) {
      const int64_t position = global_positions[lane];
      if (position >= 0 && position < local_candidate_count) {
        const int64_t local_index = local_indices[position];
        if (local_index >= 0 && local_index < chunk_size) {
          lane_token64 = (position / kTopK) * chunk_size + local_index;
        }
      }
    } else {
      lane_token64 = top_indices[lane];
    }
  }
  const bool lane_valid =
      lane < kTopK && lane_token64 >= 0 && lane_token64 < vocab_size;
  const int lane_token = lane_valid ? static_cast<int>(lane_token64) : -1;
  const float lane_raw_value = lane < kTopK ? top_values[lane] : -FLT_MAX;
  const float lane_value = isfinite(lane_raw_value) ? lane_raw_value : -FLT_MAX;
  const int valid_count = __popc(__ballot_sync(0xffffffff, lane_valid));
  int lane_rank = 0;
#pragma unroll
  for (int source = 0; source < kTopK; ++source) {
    const int other_token = __shfl_sync(0xffffffff, lane_token, source);
    const float other_value = __shfl_sync(0xffffffff, lane_value, source);
    if (lane_valid && other_token >= 0 && other_token < vocab_size &&
        top1_better(other_value, other_token, lane_value, lane_token)) {
      ++lane_rank;
    }
  }
  if (lane_valid) {
    selected_values[lane_rank] = lane_value;
    selected_ids[lane_rank] = static_cast<int64_t>(lane_token);
  } else if (lane >= valid_count && lane < kTopK) {
    // Preserve the prior fallback for malformed candidate ids.
    selected_values[lane] = -FLT_MAX;
    selected_ids[lane] = lane;
  }
  __syncwarp();

  if (lane < kTopK) {
    selected_random[lane] = native_exponential_for_index(
        selected_ids[lane], total_threads, philox_args);
  }
  __syncwarp();

  if (lane == 0) {
    const float max_value = selected_values[0];
    float probability_sum = 0.f;
    for (int index = 0; index < kTopK; ++index) {
      const float probability = __expf(selected_values[index] - max_value);
      selected_values[index] = probability;
      probability_sum += probability;
    }

    float cumulative_probability = 0.f;
    float filtered_probability_sum = 0.f;
    for (int index = 0; index < kTopK; ++index) {
      const float probability = selected_values[index] / probability_sum;
      const bool keep = index == 0 || cumulative_probability < top_p;
      cumulative_probability += probability;
      selected_values[index] = keep ? probability : 0.f;
      filtered_probability_sum += selected_values[index];
    }

    float best_score = -FLT_MAX;
    int64_t sampled_token = selected_ids[0];
    for (int index = 0; index < kTopK; ++index) {
      const float probability =
          selected_values[index] / filtered_probability_sum;
      if (EmitMetadata) {
        sparse_probs_out[index] = probability;
        sparse_ids_out[index] = selected_ids[index];
      }
      const float score = probability / selected_random[index];
      if (score > best_score) {
        best_score = score;
        sampled_token = selected_ids[index];
      }
    }
    sampled_token_out[0] = sampled_token;
  }
}

template <bool EmitMetadata, bool ChunkedIndices>
void launch_sm70_sample_sorted_top20_philox(
    torch::Tensor sampled_token_out, int64_t* sparse_ids_out,
    float* sparse_probs_out, torch::Tensor top_values,
    const int64_t* top_indices, const int64_t* local_indices,
    const int64_t* global_positions, int64_t local_candidate_count,
    int chunk_size, const std::optional<at::Generator>& generator,
    int64_t vocab_size, double top_p) {
  const auto* properties = at::cuda::getCurrentDeviceProperties();
  const int64_t blocks_per_sm =
      properties->maxThreadsPerMultiProcessor / kNativeBlockSize;
  const int64_t required_blocks =
      (vocab_size + kNativeBlockSize - 1) / kNativeBlockSize;
  const int64_t grid_blocks = std::min<int64_t>(
      required_blocks, properties->multiProcessorCount * blocks_per_sm);
  const int64_t total_threads = grid_blocks * kNativeBlockSize;
  const uint64_t counter_offset =
      ((vocab_size - 1) / (total_threads * kNativeUnroll) + 1) *
      kMaxGeneratorOffsetsPerCurandCall;

  auto* cuda_generator = at::get_generator_or_default<at::CUDAGeneratorImpl>(
      generator, at::cuda::detail::getDefaultCUDAGenerator());
  at::PhiloxCudaState philox_args;
  {
    std::lock_guard<std::mutex> lock(cuda_generator->mutex_);
    philox_args = cuda_generator->philox_cuda_state(counter_offset);
  }

  sm70_sample_sorted_top20_philox_kernel<EmitMetadata, ChunkedIndices>
      <<<1, kWarpSize, 0, at::cuda::getCurrentCUDAStream()>>>(
          sampled_token_out.data_ptr<int64_t>(), sparse_ids_out,
          sparse_probs_out, top_values.data_ptr<float>(), top_indices,
          local_indices, global_positions, local_candidate_count, chunk_size,
          vocab_size, total_threads, static_cast<float>(top_p), philox_args);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

void sm70_sample_sorted_top20_philox_out(
    torch::Tensor sampled_token_out, torch::Tensor sparse_ids_out,
    torch::Tensor sparse_probs_out, torch::Tensor top_values,
    torch::Tensor top_indices, const std::optional<at::Generator>& generator,
    int64_t vocab_size, double top_p) {
  TORCH_CHECK(sampled_token_out.is_cuda() && sparse_ids_out.is_cuda() &&
                  sparse_probs_out.is_cuda() && top_values.is_cuda() &&
                  top_indices.is_cuda(),
              "sm70_sample_sorted_top20_philox_out: tensors must be CUDA");
  TORCH_CHECK(sampled_token_out.scalar_type() == torch::kInt64 &&
                  sparse_ids_out.scalar_type() == torch::kInt64 &&
                  sparse_probs_out.scalar_type() == torch::kFloat32 &&
                  top_values.scalar_type() == torch::kFloat32 &&
                  top_indices.scalar_type() == torch::kInt64,
              "sm70_sample_sorted_top20_philox_out: dtype mismatch");
  TORCH_CHECK(sampled_token_out.numel() == 1 &&
                  sparse_ids_out.numel() == kTopK &&
                  sparse_probs_out.numel() == kTopK &&
                  top_values.numel() == kTopK && top_indices.numel() == kTopK,
              "sm70_sample_sorted_top20_philox_out: shape mismatch");
  TORCH_CHECK(
      sampled_token_out.is_contiguous() && sparse_ids_out.is_contiguous() &&
          sparse_probs_out.is_contiguous() && top_values.is_contiguous() &&
          top_indices.is_contiguous(),
      "sm70_sample_sorted_top20_philox_out: tensors must be contiguous");
  TORCH_CHECK(vocab_size >= kTopK && top_p > 0.0 && top_p <= 1.0,
              "sm70_sample_sorted_top20_philox_out: invalid parameters");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(top_values));
  launch_sm70_sample_sorted_top20_philox<true, false>(
      sampled_token_out, sparse_ids_out.data_ptr<int64_t>(),
      sparse_probs_out.data_ptr<float>(), top_values,
      top_indices.data_ptr<int64_t>(), nullptr, nullptr, 0, 0, generator,
      vocab_size, top_p);
}

void sm70_sample_sorted_top20_philox_token_out(
    torch::Tensor sampled_token_out, torch::Tensor top_values,
    torch::Tensor top_indices, const std::optional<at::Generator>& generator,
    int64_t vocab_size, double top_p) {
  TORCH_CHECK(
      sampled_token_out.is_cuda() && top_values.is_cuda() &&
          top_indices.is_cuda(),
      "sm70_sample_sorted_top20_philox_token_out: tensors must be CUDA");
  TORCH_CHECK(sampled_token_out.scalar_type() == torch::kInt64 &&
                  top_values.scalar_type() == torch::kFloat32 &&
                  top_indices.scalar_type() == torch::kInt64,
              "sm70_sample_sorted_top20_philox_token_out: dtype mismatch");
  TORCH_CHECK(sampled_token_out.numel() == 1 && top_values.numel() == kTopK &&
                  top_indices.numel() == kTopK,
              "sm70_sample_sorted_top20_philox_token_out: shape mismatch");
  TORCH_CHECK(
      sampled_token_out.is_contiguous() && top_values.is_contiguous() &&
          top_indices.is_contiguous(),
      "sm70_sample_sorted_top20_philox_token_out: tensors must be contiguous");
  TORCH_CHECK(vocab_size >= kTopK && top_p > 0.0 && top_p <= 1.0,
              "sm70_sample_sorted_top20_philox_token_out: invalid parameters");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(top_values));
  launch_sm70_sample_sorted_top20_philox<false, false>(
      sampled_token_out, nullptr, nullptr, top_values,
      top_indices.data_ptr<int64_t>(), nullptr, nullptr, 0, 0, generator,
      vocab_size, top_p);
}

void sm70_sample_chunked_top20_philox_token_out(
    torch::Tensor sampled_token_out, torch::Tensor global_values,
    torch::Tensor local_indices, torch::Tensor global_positions,
    const std::optional<at::Generator>& generator, int64_t vocab_size,
    double top_p, int64_t chunk_size) {
  TORCH_CHECK(
      sampled_token_out.is_cuda() && global_values.is_cuda() &&
          local_indices.is_cuda() && global_positions.is_cuda(),
      "sm70_sample_chunked_top20_philox_token_out: tensors must be CUDA");
  TORCH_CHECK(sampled_token_out.scalar_type() == torch::kInt64 &&
                  global_values.scalar_type() == torch::kFloat32 &&
                  local_indices.scalar_type() == torch::kInt64 &&
                  global_positions.scalar_type() == torch::kInt64,
              "sm70_sample_chunked_top20_philox_token_out: dtype mismatch");
  TORCH_CHECK(sampled_token_out.numel() == 1 &&
                  global_values.numel() == kTopK && local_indices.dim() == 2 &&
                  local_indices.size(1) == kTopK &&
                  global_positions.numel() == kTopK,
              "sm70_sample_chunked_top20_philox_token_out: shape mismatch");
  TORCH_CHECK(
      sampled_token_out.is_contiguous() && global_values.is_contiguous() &&
          local_indices.is_contiguous() && global_positions.is_contiguous(),
      "sm70_sample_chunked_top20_philox_token_out: tensors must be contiguous");
  TORCH_CHECK(chunk_size > 0 &&
                  local_indices.size(0) * chunk_size == vocab_size &&
                  vocab_size >= kTopK && top_p > 0.0 && top_p <= 1.0,
              "sm70_sample_chunked_top20_philox_token_out: invalid parameters");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(global_values));
  launch_sm70_sample_sorted_top20_philox<false, true>(
      sampled_token_out, nullptr, nullptr, global_values, nullptr,
      local_indices.data_ptr<int64_t>(), global_positions.data_ptr<int64_t>(),
      local_indices.numel(), static_cast<int>(chunk_size), generator,
      vocab_size, top_p);
}
