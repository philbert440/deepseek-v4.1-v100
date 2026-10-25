// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// DS-V4.1 fused sparse PREFILL attention for SM70 (V100), VLLM_DS41_ATTN_FUSED_PREFILL.
//
// Per query token t (one CTA per (t, 16-head group)): the 16 local heads attend over <= NW window rows (fp16 SWA
// cache records) + <= NC compressed rows (packed FP4 E2M1 x E4M3/16 records, 288 B, unpacked in-kernel with the
// qat math), row ids by index (-1 = invalid, -> +0.0 row, score -inf), plus the per-head attention sink in the
// denominator only. K = V = the 512-wide latent (MQA): every K/V block is loaded once and used by all 16 heads,
// which are the WMMA M dimension (16 rows).
//
//   s_j = scale * q . k_j (FP32)    m = max(-1e30, max_j s_j)    o = sum_j exp(s_j - m) k_j / (sum_j exp(s_j - m)
//                                                                                         + exp(sink - m))
//
// Tiling: 4 warps; warp w owns latent dims [128w, 128w + 128) for BOTH products, so it gathers (and unpacks) only
// its own 128-dim slice of each 32-key block into a warp-private smem slice (no CTA barrier around the gather):
//   QK : S_w[16 x 32] = Q[:, slice_w] . K[32, slice_w]^T     (Q fragments held in registers for the whole loop)
//   CTA: S = sum_w S_w (fixed order) -> online softmax in FP32 (running max floored at -1e30, as the reference)
//   PV : O[:, slice_w] (registers, 8 accumulator fragments) = alpha * O + P[16 x 32] . V[32, slice_w]
// P is fed to the tensor cores as fp16 hi + fp16 lo (P_SPLIT, default) so the product carries ~22 bits of P
// (the cuBLAS bmm path rounds P to fp16 once); the denominator sums the FP32 p. Two __syncthreads per block.
// A block whose 32 row ids are all -1 is skipped (CTA-uniform: every warp reads the same ids).
//
// Resources (sm_70): 128 threads, 46,912 B static-size dynamic smem (< 48 KB, no opt-in), __launch_bounds__(128, 2)
// -> 2 CTAs / SM (8 warps). No workspace, no allocation: CUDA-graph safe.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cmath>
#include <cstdint>

namespace ds41_sparse_prefill {
using namespace nvcuda;

constexpr int kD = 512;                 // latent width (K = V)
constexpr int kHG = 16;                 // heads per CTA = WMMA M
constexpr int kWarps = 4;
constexpr int kThreads = kWarps * 32;
constexpr int kDW = kD / kWarps;        // 128 latent dims per warp
constexpr int kBN = 32;                 // keys per block
constexpr int kPacked = kD / 2 + kD / 16;   // 288 B per packed record
constexpr int kKT = kDW / 16;           // QK k-steps per warp
constexpr int kNT = kDW / 16;           // PV n-tiles per warp
constexpr int kLdK = kDW + 8;           // half, per-warp K/V slice row
constexpr int kLdS = kBN + 4;           // float, partial scores
constexpr int kLdP = kBN + 8;           // half, probabilities
constexpr int kLdO = kDW + 4;           // float, output staging (aliases the warp's K slice)
constexpr float kRunningMaxFloor = -1e30f;

constexpr int kSliceBytes = kBN * kLdK * 2;
constexpr int kOffK = 0;
constexpr int kOffS = kOffK + kWarps * kSliceBytes;
constexpr int kOffP = kOffS + kWarps * kHG * kLdS * 4;
constexpr int kOffPlo = kOffP + kHG * kLdP * 2;
constexpr int kOffM = kOffPlo + kHG * kLdP * 2;
constexpr int kOffL = kOffM + kHG * 4;
constexpr int kOffA = kOffL + kHG * 4;
constexpr int kOffValid = kOffA + kHG * 4;
constexpr int kSmemBytes = kOffValid + kBN * 4;
static_assert(kOffS % 32 == 0 && kOffP % 32 == 0 && kOffPlo % 32 == 0 && kSliceBytes % 32 == 0,
              "WMMA pointers need 256-bit alignment");
static_assert(kHG * kLdO * 4 <= kSliceBytes, "output staging must fit in the warp's K slice");
static_assert(kHG * 16 * 4 <= kWarps * kHG * kLdS * 4, "row-id table must fit in the score area");
static_assert(kSmemBytes <= 48 * 1024, "keep under the default 48 KB so 2 CTAs fit per SM");

using FragA = wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major>;
using FragBc = wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major>;
using FragBr = wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major>;
using FragC = wmma::fragment<wmma::accumulator, 16, 16, 16, float>;

// qat.e2m1_value: 4-bit code -> FP32 (exact), sign from bit 3 (keeps -0.0).
__device__ __forceinline__ float e2m1_value(uint32_t code) {
  const uint32_t m = code & 7u;
  const float v = m < 4u ? 0.5f * static_cast<float>(m) : (m == 7u ? 6.0f : static_cast<float>(m - 2u));
  return __uint_as_float(__float_as_uint(v) | ((code & 8u) << 28));
}

// qat.e4m3_value: E4M3FN byte (sign ignored) -> FP32 (exact); 0x7F -> NaN.
__device__ __forceinline__ float e4m3_value(uint32_t b) {
  const uint32_t e = (b >> 3) & 0xFu, m = b & 7u;
  const float v = e == 0u ? static_cast<float>(m) * 0.001953125f       // m * 2^-9
                          : __uint_as_float(((e + 120u) << 23) | (m << 20));
  return (b & 0x7Fu) == 0x7Fu ? __int_as_float(0x7fc00000) : v;
}

__device__ __forceinline__ uint32_t pack_half2(float a, float b) {
  const half2 h = __halves2half2(__float2half_rn(a), __float2half_rn(b));
  return *reinterpret_cast<const uint32_t*>(&h);
}

template <bool kPSplit>
__global__ void __launch_bounds__(kThreads, 2) ds41_sparse_prefill_kernel(
    const half* __restrict__ q, int64_t q_st, int64_t q_sh,
    const half* __restrict__ swa, const int64_t* __restrict__ widx, int64_t widx_st, int nw,
    const uint8_t* __restrict__ ckv, const int64_t* __restrict__ cidx, int64_t cidx_st, int nc,
    const float* __restrict__ sink, float scale,
    half* __restrict__ out, int64_t o_st, int64_t o_sh) {
  extern __shared__ __align__(128) unsigned char smem[];
  const int t = blockIdx.x;
  const int h0 = blockIdx.y * kHG;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  half* sK = reinterpret_cast<half*>(smem + kOffK + warp * kSliceBytes);
  float* sS = reinterpret_cast<float*>(smem + kOffS);
  half* sP = reinterpret_cast<half*>(smem + kOffP);
  half* sPlo = reinterpret_cast<half*>(smem + kOffPlo);
  float* sM = reinterpret_cast<float*>(smem + kOffM);
  float* sL = reinterpret_cast<float*>(smem + kOffL);
  float* sA = reinterpret_cast<float*>(smem + kOffA);
  int* sValid = reinterpret_cast<int*>(smem + kOffValid);
  const unsigned full = 0xffffffffu;

  // Row index of every accumulator element: load a [16 x 16] table of row ids through the same fragment type
  // (element <-> (row, col) mapping is identical for identical fragment types; no layout assumption).
  for (int i = tid; i < kHG * 16; i += kThreads) sS[i] = static_cast<float>(i / 16);
  if (tid < kHG) {
    sM[tid] = kRunningMaxFloor;
    sL[tid] = 0.0f;
  }
  __syncthreads();
  FragC rowf;
  wmma::load_matrix_sync(rowf, sS, 16, wmma::mem_row_major);

  FragA qa[kKT];
  const half* qb = q + static_cast<int64_t>(t) * q_st + static_cast<int64_t>(h0) * q_sh + warp * kDW;
#pragma unroll
  for (int kk = 0; kk < kKT; ++kk) wmma::load_matrix_sync(qa[kk], qb + kk * 16, static_cast<unsigned>(q_sh));
  FragC o[kNT];
#pragma unroll
  for (int nt = 0; nt < kNT; ++nt) wmma::fill_fragment(o[nt], 0.0f);
  __syncthreads();                                   // every warp has its row table before sS is reused

  const int nbw = (nw + kBN - 1) / kBN, nbc = (nc + kBN - 1) / kBN;
  const int64_t* wrow = widx + static_cast<int64_t>(t) * widx_st;
  const int64_t* crow = cidx + static_cast<int64_t>(t) * cidx_st;
  const int nb = nbw + nbc;
  auto row_of = [&](int b) -> long long {                        // lane j: row id of key j of block b
    if (b >= nb) return -1;
    const bool pk = b >= nbw;
    const int k = (pk ? b - nbw : b) * kBN + lane;
    if (k >= (pk ? nc : nw)) return -1;
    return pk ? crow[k] : wrow[k];
  };
  long long row_next = row_of(0);
  for (int b = 0; b < nb; ++b) {
    const bool packed = b >= nbw;
    const long long row = row_next;
    row_next = row_of(b + 1);                                    // prefetch: consumed one block later
    if (__ballot_sync(full, row >= 0) == 0u) continue;          // uniform across the CTA
    if (warp == 0) sValid[lane] = row >= 0 ? 1 : 0;

    // ---- gather this warp's 128-dim slice of the 32 rows (invalid rows -> +0.0) ----
    if (!packed) {
      const int sub = lane & 15, rsel = lane >> 4;               // 16 lanes x 16 B per row, 2 rows per pass
#pragma unroll
      for (int p0 = 0; p0 < 16; p0 += 4) {
        uint4 v[4];
#pragma unroll
        for (int u = 0; u < 4; ++u) {
          const long long r = __shfl_sync(full, row, (p0 + u) * 2 + rsel);
          v[u] = make_uint4(0u, 0u, 0u, 0u);
          if (r >= 0) v[u] = __ldg(reinterpret_cast<const uint4*>(swa + r * kD + warp * kDW + sub * 8));
        }
#pragma unroll
        for (int u = 0; u < 4; ++u)
          *reinterpret_cast<uint4*>(sK + ((p0 + u) * 2 + rsel) * kLdK + sub * 8) = v[u];
      }
    } else {
      const int bb = lane & 7, jq = lane >> 3;                   // (key, 16-dim scale block): 8 lanes per row
      uint2 nib[8];
      uint32_t sb[8];
      bool ok[8];
#pragma unroll
      for (int p = 0; p < 8; ++p) {
        const long long r = __shfl_sync(full, row, p * 4 + jq);
        ok[p] = r >= 0;
        nib[p] = make_uint2(0u, 0u);
        sb[p] = 0u;
        if (ok[p]) {
          const uint8_t* base = ckv + r * kPacked;
          nib[p] = __ldg(reinterpret_cast<const uint2*>(base + warp * (kDW / 2) + bb * 8));
          sb[p] = __ldg(base + kD / 2 + warp * (kDW / 16) + bb);
        }
      }
#pragma unroll
      for (int p = 0; p < 8; ++p) {
        const float sc = e4m3_value(sb[p]);
        uint32_t h[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const uint32_t byte = ((i < 4 ? nib[p].x : nib[p].y) >> (8 * (i & 3))) & 0xFFu;
          // byte i of the block holds dims 2i (low nibble) and 2i + 1 (high), as qat._unpack_ckv_kernel
          h[i] = ok[p] ? pack_half2(e2m1_value(byte & 0xFu) * sc, e2m1_value(byte >> 4) * sc) : 0u;
        }
        uint4* dst = reinterpret_cast<uint4*>(sK + (p * 4 + jq) * kLdK + bb * 16);
        dst[0] = make_uint4(h[0], h[1], h[2], h[3]);
        dst[1] = make_uint4(h[4], h[5], h[6], h[7]);
      }
    }
    __syncwarp();

    // ---- partial scores over this warp's dims ----
#pragma unroll
    for (int nt = 0; nt < kBN / 16; ++nt) {
      FragC s;
      wmma::fill_fragment(s, 0.0f);
#pragma unroll
      for (int kk = 0; kk < kKT; ++kk) {
        FragBc kf;
        wmma::load_matrix_sync(kf, sK + nt * 16 * kLdK + kk * 16, kLdK);
        wmma::mma_sync(s, qa[kk], kf, s);
      }
      wmma::store_matrix_sync(sS + warp * kHG * kLdS + nt * 16, s, kLdS, wmma::mem_row_major);
    }
    __syncthreads();

    // ---- online softmax: 8 threads per head row, 4 keys each ----
    {
      const int r = tid >> 3, c0 = (tid & 7) * 4;
      float sv[4];
      float mx = -INFINITY;
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        const int c = c0 + u;
        float a = 0.0f;
#pragma unroll
        for (int w = 0; w < kWarps; ++w) a += sS[w * kHG * kLdS + r * kLdS + c];
        sv[u] = sValid[c] ? a * scale : -INFINITY;
        mx = fmaxf(mx, sv[u]);
      }
#pragma unroll
      for (int off = 4; off > 0; off >>= 1) mx = fmaxf(mx, __shfl_xor_sync(full, mx, off));
      const float m_old = sM[r];
      const float m_new = fmaxf(m_old, mx);                     // finite: m_old >= -1e30
      float sum = 0.0f;
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        const float p = expf(sv[u] - m_new);                    // exp(-inf) = 0 for invalid keys
        sum += p;
        const half ph = __float2half_rn(p);
        sP[r * kLdP + c0 + u] = ph;
        if (kPSplit) sPlo[r * kLdP + c0 + u] = __float2half_rn(p - __half2float(ph));
      }
#pragma unroll
      for (int off = 4; off > 0; off >>= 1) sum += __shfl_xor_sync(full, sum, off);
      __syncwarp();
      if ((tid & 7) == 0) {
        const float alpha = expf(m_old - m_new);
        sL[r] = sL[r] * alpha + sum;
        sM[r] = m_new;
        sA[r] = alpha;
      }
    }
    __syncthreads();

    // ---- O = alpha * O + P V over this warp's dims ----
#pragma unroll
    for (int nt = 0; nt < kNT; ++nt)
#pragma unroll
      for (int i = 0; i < o[nt].num_elements; ++i) o[nt].x[i] *= sA[static_cast<int>(rowf.x[i])];
#pragma unroll
    for (int kk = 0; kk < kBN / 16; ++kk) {
      FragA pa, pl;
      wmma::load_matrix_sync(pa, sP + kk * 16, kLdP);
      if (kPSplit) wmma::load_matrix_sync(pl, sPlo + kk * 16, kLdP);
#pragma unroll
      for (int nt = 0; nt < kNT; ++nt) {
        FragBr vf;
        wmma::load_matrix_sync(vf, sK + kk * 16 * kLdK + nt * 16, kLdK);
        wmma::mma_sync(o[nt], pa, vf, o[nt]);
        if (kPSplit) wmma::mma_sync(o[nt], pl, vf, o[nt]);
      }
    }
  }

  // ---- epilogue: denominator with the sink, normalise, fp16 out ----
  if (tid < kHG) sL[tid] = sL[tid] + expf(sink[h0 + tid] - sM[tid]);
  __syncthreads();
  float* stage = reinterpret_cast<float*>(sK);
#pragma unroll
  for (int nt = 0; nt < kNT; ++nt) {
#pragma unroll
    for (int i = 0; i < o[nt].num_elements; ++i) o[nt].x[i] = o[nt].x[i] / sL[static_cast<int>(rowf.x[i])];
    wmma::store_matrix_sync(stage + nt * 16, o[nt], kLdO, wmma::mem_row_major);
  }
  __syncwarp();
  half* ob = out + static_cast<int64_t>(t) * o_st + static_cast<int64_t>(h0) * o_sh + warp * kDW;
#pragma unroll
  for (int it = 0; it < kHG * kDW / 8 / 32; ++it) {
    const int g = it * 32 + lane, r = g / (kDW / 8), c8 = (g % (kDW / 8)) * 8;
    const float* s = stage + r * kLdO + c8;
    *reinterpret_cast<uint4*>(ob + r * o_sh + c8) =
        make_uint4(pack_half2(s[0], s[1]), pack_half2(s[2], s[3]), pack_half2(s[4], s[5]), pack_half2(s[6], s[7]));
  }
}

template <bool kPSplit>
void launch(const at::Tensor& q, const at::Tensor& swa, const at::Tensor& widx, const at::Tensor& ckv,
            const at::Tensor& cidx, const at::Tensor& sink, double scale, at::Tensor& out, cudaStream_t stream) {
  static bool configured = false;
  if (!configured) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(ds41_sparse_prefill_kernel<kPSplit>,
                                        cudaFuncAttributePreferredSharedMemoryCarveout,
                                        cudaSharedmemCarveoutMaxShared));
    configured = true;
  }
  const int64_t T = q.size(0), H = q.size(1);
  const int nc = ckv.defined() && ckv.numel() ? static_cast<int>(cidx.size(1)) : 0;
  const dim3 grid(static_cast<unsigned>(T), static_cast<unsigned>(H / kHG));
  ds41_sparse_prefill_kernel<kPSplit><<<grid, kThreads, kSmemBytes, stream>>>(
      reinterpret_cast<const half*>(q.data_ptr()), q.stride(0), q.stride(1),
      reinterpret_cast<const half*>(swa.data_ptr()), widx.data_ptr<int64_t>(), widx.stride(0),
      static_cast<int>(widx.size(1)),
      nc ? ckv.data_ptr<uint8_t>() : nullptr, nc ? cidx.data_ptr<int64_t>() : nullptr, nc ? cidx.stride(0) : 0, nc,
      sink.data_ptr<float>(), static_cast<float>(scale),
      reinterpret_cast<half*>(out.data_ptr()), out.stride(0), out.stride(1));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

static bool aligned(const at::Tensor& x, int bytes) {
  return reinterpret_cast<uintptr_t>(x.data_ptr()) % bytes == 0;
}

// q [T, H, 512] fp16; swa [N, 512] fp16 + widx [T, NW] int64; ckv [M, 288] uint8 + cidx [T, NC] int64 (or empty
// tensors for a window-only layer); sink [>= H] fp32; out [T, H, 512] fp16 (may be a row slice of a larger buffer).
void fwd(const at::Tensor& q, const at::Tensor& swa, const at::Tensor& widx, const at::Tensor& ckv,
         const at::Tensor& cidx, const at::Tensor& sink, double scale, at::Tensor& out, bool p_split) {
  TORCH_CHECK(q.is_cuda() && q.scalar_type() == at::kHalf && q.dim() == 3 && q.size(2) == kD,
              "q must be CUDA fp16 [T, H, 512]");
  TORCH_CHECK(q.size(1) % kHG == 0, "local heads must be a multiple of 16, got ", q.size(1));
  TORCH_CHECK(q.stride(2) == 1 && q.stride(1) % 16 == 0 && q.stride(0) % 16 == 0 && aligned(q, 32),
              "q rows must be contiguous, 32-byte aligned, strides multiples of 16 elements");
  TORCH_CHECK(swa.is_cuda() && swa.scalar_type() == at::kHalf && swa.dim() == 2 && swa.size(1) == kD &&
                  swa.is_contiguous() && aligned(swa, 16),
              "window rows must be contiguous fp16 [N, 512]");
  const int64_t T = q.size(0);
  TORCH_CHECK(widx.is_cuda() && widx.scalar_type() == at::kLong && widx.dim() == 2 && widx.size(0) == T &&
                  widx.stride(1) == 1,
              "window ids must be int64 [T, NW] with unit inner stride");
  if (ckv.defined() && ckv.numel()) {
    TORCH_CHECK(ckv.is_cuda() && ckv.scalar_type() == at::kByte && ckv.dim() == 2 && ckv.size(1) == kPacked &&
                    ckv.is_contiguous() && aligned(ckv, 8),
                "compressed rows must be contiguous packed uint8 [M, 288]");
    TORCH_CHECK(cidx.is_cuda() && cidx.scalar_type() == at::kLong && cidx.dim() == 2 && cidx.size(0) == T &&
                    cidx.stride(1) == 1,
                "compressed ids must be int64 [T, NC] with unit inner stride");
  }
  TORCH_CHECK(sink.is_cuda() && sink.scalar_type() == at::kFloat && sink.is_contiguous() &&
                  sink.numel() >= q.size(1),
              "sink must be contiguous fp32 [>= H]");
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == at::kHalf && out.sizes() == q.sizes() && out.stride(2) == 1 &&
                  out.stride(1) % 8 == 0 && out.stride(0) % 8 == 0 && aligned(out, 16),
              "out must be fp16 shaped like q, 16-byte aligned rows");
  if (T == 0) return;
  const at::cuda::OptionalCUDAGuard guard(q.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (p_split)
    launch<true>(q, swa, widx, ckv, cidx, sink, scale, out, stream);
  else
    launch<false>(q, swa, widx, ckv, cidx, sink, scale, out, stream);
}

}  // namespace ds41_sparse_prefill

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fwd", &ds41_sparse_prefill::fwd, "DS-V4.1 fused sparse prefill attention (SM70 WMMA, FP32 online softmax)");
  m.attr("latent_dim") = ds41_sparse_prefill::kD;
  m.attr("heads_per_cta") = ds41_sparse_prefill::kHG;
  m.attr("keys_per_block") = ds41_sparse_prefill::kBN;
  m.attr("smem_bytes") = ds41_sparse_prefill::kSmemBytes;
  m.attr("version") = 1;
}
