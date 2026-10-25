// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// DS-V4.1 fused lightning-indexer scoring + top-512 for SM70 (V100), VLLM_DS41_INDEXER_FUSED.
//
//   score[t, j] = sum_h relu(q[t, h] . k_j) * w[t, h]        (q [T, 32, 128] fp16, k fp16, FP32 out)
//
// * cand_scores: reindex layers (24-36). One CTA per (query row t, 2048-key split of its 16,384 candidate slots).
//   Slot c of row t is position cand[t, c / 8] * 8 + c % 8 (cand row ascending, -1 pad; row[0] == CAND_ALL ->
//   block c / 8); invalid slots (block -1 or position >= end) score -inf. Keys are gathered straight from the
//   index-K cache through the request's block table (row = bt[req, p / S] * S + p % S): no [T, 16K, 128] tensor.
// * dense_scores: the full scan (layer 20 / scan+mask): keys [n, 128] contiguous, one CTA per (4 query rows,
//   key split); every 64-key chunk is loaded once for the 4 rows. Writes out[t, j] for j < n (ld = out stride).
// * topk512: per row, the 512 largest finite scores (ties -> the lowest column = the lowest position, since columns
//   are position-ascending in both layouts), written ascending by position, -1 padded; -inf never selected.
//   Radix select (4 x 8-bit histograms) + one ordered compaction pass (cub BlockScan).
//
// Scores: WMMA (HMMA.884) m16n16k16, FP32 accumulate, the row's 32 heads are M (2 tiles), 16 keys per n-tile,
// K = 128 in 8 k-steps; q fragments stay in registers for the whole CTA. Epilogue per n-tile through a warp-private
// smem tile: relu, then s = s + relu(l) * w[h] for h = 0..31 in order with explicit RN mul / add (no FMA), the same
// elementwise ops as the torch path ``(relu(logits) * w).sum(1)``. Not bit-identical to cuBLAS (accumulation order
// inside the dot products and the head sum may differ): gate on top-512 SET equality + score drift.
//
// Resources: 128 threads, ~28 KB static smem, __launch_bounds__(128, 2). Scratch scores are allocated by the caller.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cub/cub.cuh>
#include <cstdint>

namespace ds41_indexer_fused {
using namespace nvcuda;

constexpr int kH = 32;                 // indexer heads (WMMA M = 2 x 16)
constexpr int kD = 128;                // index-key dim (K = 8 x 16)
constexpr int kKT = kD / 16;
constexpr int kBN = 64;                // keys per chunk (4 n-tiles of 16)
constexpr int kWarps = 4;
constexpr int kThreads = kWarps * 32;
constexpr int kLdK = kD + 8;           // half, smem key row
constexpr int kLdS = 16 + 4;           // float, warp score tile
constexpr int kTopK = 512;
constexpr int kCandBlock = 8;
constexpr int kCandSlots = 16384;      // 2048 blocks x 8
constexpr int kCandAll = -2;
constexpr int kU = kBN * kD / 8 / kThreads;   // uint4 loads per thread per chunk (8)
constexpr int kTopThreads = 256;

using FragA = wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major>;
using FragB = wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major>;
using FragC = wmma::fragment<wmma::accumulator, 16, 16, 16, float>;

struct __align__(32) Smem {
  half k[kBN * kLdK];                  // 17,408 B: the chunk's keys, row-major [key][dim]
  float s[kWarps][kH * kLdS];          // 10,240 B: per-warp [head][key] logits of one n-tile
  float w[kWarps][kH];                 // per-warp row weights
};

__device__ __forceinline__ void load_q(FragA (&qa)[2][kKT], const half* qrow, int ldh) {
#pragma unroll
  for (int m = 0; m < 2; ++m)
#pragma unroll
    for (int kk = 0; kk < kKT; ++kk) wmma::load_matrix_sync(qa[m][kk], qrow + m * 16 * ldh + kk * 16, ldh);
}

// logits of the warp's row (32 heads) x keys [16 nt, 16 nt + 16) of the smem chunk -> lane l < 16 returns the
// score of key 16 nt + l.
__device__ __forceinline__ float tile_score(const FragA (&qa)[2][kKT], const half* sK, int nt, float* sS,
                                            const float* w, int lane) {
  FragC acc0, acc1;
  wmma::fill_fragment(acc0, 0.0f);
  wmma::fill_fragment(acc1, 0.0f);
#pragma unroll
  for (int kk = 0; kk < kKT; ++kk) {
    FragB kb;
    wmma::load_matrix_sync(kb, sK + nt * 16 * kLdK + kk * 16, kLdK);
    wmma::mma_sync(acc0, qa[0][kk], kb, acc0);
    wmma::mma_sync(acc1, qa[1][kk], kb, acc1);
  }
  wmma::store_matrix_sync(sS, acc0, kLdS, wmma::mem_row_major);
  wmma::store_matrix_sync(sS + 16 * kLdS, acc1, kLdS, wmma::mem_row_major);
  __syncwarp();
  float s = 0.0f;
  if (lane < 16) {
#pragma unroll
    for (int h = 0; h < kH; ++h) {
      float v = sS[h * kLdS + lane];
      v = v > 0.0f ? v : 0.0f;
      s = __fadd_rn(s, __fmul_rn(v, w[h]));
    }
  }
  __syncwarp();
  return s;
}

__device__ __forceinline__ int64_t cand_pos(const int* crow, bool all, int c, int64_t end) {
  const int blk = all ? (c >> 3) : __ldg(crow + (c >> 3));
  if (blk < 0) return -1;
  const int64_t p = static_cast<int64_t>(blk) * kCandBlock + (c & 7);
  return p < end ? p : -1;
}

// ------------------------------------------------------------------------------------------- candidate scores
__global__ void __launch_bounds__(kThreads, 2)
cand_scores_kernel(const half* __restrict__ q, int64_t q_s0, int q_s1, const float* __restrict__ w,
                   const int* __restrict__ cand, int64_t cand_s0, const int64_t* __restrict__ ends,
                   const half* __restrict__ keys, const int* __restrict__ tok2req, const int* __restrict__ bt,
                   int64_t bt_s0, int storage, float* __restrict__ scores, int slots_per_cta) {
  __shared__ Smem sm;
  const int t = blockIdx.x;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int c_begin = blockIdx.y * slots_per_cta;
  const int c_end = min(kCandSlots, c_begin + slots_per_cta);
  const int64_t end = ends[t];
  const int* crow = cand + t * cand_s0;
  const bool all = __ldg(crow) == kCandAll;
  const int* btrow = bt + static_cast<int64_t>(__ldg(tok2req + t)) * bt_s0;
  if (tid < kH) sm.w[0][tid] = w[static_cast<int64_t>(t) * kH + tid];
  FragA qa[2][kKT];
  load_q(qa, q + t * q_s0, q_s1);
  float* srow = scores + static_cast<int64_t>(t) * kCandSlots;

  uint4 v[kU];
  bool any_next = false;
  auto fetch = [&](int c0) {
    any_next = false;
#pragma unroll
    for (int u = 0; u < kU; ++u) {
      const int idx = u * kThreads + tid, key = idx >> 4, part = idx & 15, c = c0 + key;
      const int64_t p = c < c_end ? cand_pos(crow, all, c, end) : -1;
      if (p >= 0) {
        const int64_t row = static_cast<int64_t>(__ldg(btrow + p / storage)) * storage + p % storage;
        v[u] = __ldg(reinterpret_cast<const uint4*>(keys + row * kD) + part);
        any_next = true;
      } else {
        v[u] = make_uint4(0, 0, 0, 0);
      }
    }
  };
  fetch(c_begin);
  for (int c0 = c_begin; c0 < c_end; c0 += kBN) {
    __syncthreads();                                  // previous chunk's fragments are consumed
#pragma unroll
    for (int u = 0; u < kU; ++u) {
      const int idx = u * kThreads + tid, key = idx >> 4, part = idx & 15;
      *reinterpret_cast<uint4*>(sm.k + key * kLdK + part * 8) = v[u];
    }
    const bool live = __syncthreads_or(any_next);
    if (c0 + kBN < c_end) fetch(c0 + kBN);           // in flight while this chunk is scored
    const int c = c0 + warp * 16 + lane;
    if (!live) {
      if (lane < 16 && c < c_end) srow[c] = -INFINITY;
      continue;
    }
    const float s = tile_score(qa, sm.k, warp, sm.s[warp], sm.w[0], lane);
    if (lane < 16 && c < c_end) srow[c] = cand_pos(crow, all, c, end) >= 0 ? s : -INFINITY;
  }
}

// ------------------------------------------------------------------------------------------- dense scores
__global__ void __launch_bounds__(kThreads, 2)
dense_scores_kernel(const half* __restrict__ q, int64_t q_s0, int q_s1, const float* __restrict__ w,
                    const half* __restrict__ keys, int n, int Q, float* __restrict__ out, int64_t ld,
                    int keys_per_cta) {
  __shared__ Smem sm;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int t = blockIdx.x * kWarps + warp;
  const bool row_ok = t < Q;
  const int j_begin = blockIdx.y * keys_per_cta;
  const int j_end = min(n, j_begin + keys_per_cta);
  if (lane < kH) sm.w[warp][lane] = row_ok ? w[static_cast<int64_t>(t) * kH + lane] : 0.0f;
  FragA qa[2][kKT];
  if (row_ok) load_q(qa, q + t * q_s0, q_s1);
  float* orow = out + static_cast<int64_t>(row_ok ? t : 0) * ld;

  uint4 v[kU];
  auto fetch = [&](int j0) {
#pragma unroll
    for (int u = 0; u < kU; ++u) {
      const int idx = u * kThreads + tid, key = idx >> 4, part = idx & 15, j = j0 + key;
      v[u] = j < j_end ? __ldg(reinterpret_cast<const uint4*>(keys + static_cast<int64_t>(j) * kD) + part)
                       : make_uint4(0, 0, 0, 0);
    }
  };
  fetch(j_begin);
  for (int j0 = j_begin; j0 < j_end; j0 += kBN) {
    __syncthreads();
#pragma unroll
    for (int u = 0; u < kU; ++u) {
      const int idx = u * kThreads + tid, key = idx >> 4, part = idx & 15;
      *reinterpret_cast<uint4*>(sm.k + key * kLdK + part * 8) = v[u];
    }
    __syncthreads();
    if (j0 + kBN < j_end) fetch(j0 + kBN);
    if (!row_ok) continue;
#pragma unroll
    for (int nt = 0; nt < kBN / 16; ++nt) {
      const float s = tile_score(qa, sm.k, nt, sm.s[warp], sm.w[warp], lane);
      const int j = j0 + nt * 16 + lane;
      if (lane < 16 && j < j_end) orow[j] = s;
    }
  }
}

// ------------------------------------------------------------------------------------------- top-512
__device__ __forceinline__ unsigned order_key(float f) {
  if (f == -INFINITY) return 0u;                    // never selected
  const unsigned b = f == 0.0f ? 0u : __float_as_uint(f);   // -0 == +0: equal scores tie by position
  return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

template <bool kCand>
__global__ void __launch_bounds__(kTopThreads)
topk512_kernel(const float* __restrict__ scores, int64_t ld, int L, const int* __restrict__ cand, int64_t cand_s0,
               int* __restrict__ out, int64_t out_s0) {
  using Scan = cub::BlockScan<int, kTopThreads>;
  __shared__ typename Scan::TempStorage tmp;
  __shared__ int hist[256];
  __shared__ unsigned s_prefix;
  __shared__ int s_k;
  const int t = blockIdx.x, tid = threadIdx.x;
  const float* row = scores + static_cast<int64_t>(t) * ld;
  int* orow = out + static_cast<int64_t>(t) * out_s0;
  unsigned thr = 0u;
  int need_eq = 0;
  if (L > kTopK) {
    unsigned prefix = 0u, mask = 0u;
    int k = kTopK;
    for (int shift = 24; shift >= 0; shift -= 8) {
      for (int i = tid; i < 256; i += kTopThreads) hist[i] = 0;
      __syncthreads();
      for (int j = tid; j < L; j += kTopThreads) {
        const unsigned key = order_key(row[j]);
        if ((key & mask) == prefix) atomicAdd(&hist[(key >> shift) & 255u], 1);
      }
      __syncthreads();
      if (tid == 0) {
        int cum = 0, b = 255;
        for (; b > 0; --b) {
          if (cum + hist[b] >= k) break;
          cum += hist[b];
        }
        s_prefix = prefix | (static_cast<unsigned>(b) << shift);
        s_k = k - cum;
      }
      __syncthreads();
      prefix = s_prefix;
      k = s_k;
      mask |= 0xFFu << shift;
      __syncthreads();
    }
    thr = prefix;
    need_eq = thr == 0u ? 0 : k;
  }
  const int* crow = kCand ? cand + t * cand_s0 : nullptr;
  const bool all = kCand && __ldg(crow) == kCandAll;
  int run_eq = 0, run_sel = 0;
  for (int base = 0; base < L; base += kTopThreads) {
    const int j = base + tid;
    const unsigned key = j < L ? order_key(row[j]) : 0u;
    const int is_gt = j < L && key > thr;
    const int is_eq = j < L && thr != 0u && key == thr;
    int eq_rank, eq_tot;
    Scan(tmp).ExclusiveSum(is_eq, eq_rank, eq_tot);
    __syncthreads();
    const int sel = is_gt || (is_eq && run_eq + eq_rank < need_eq);
    int sel_rank, sel_tot;
    Scan(tmp).ExclusiveSum(sel, sel_rank, sel_tot);
    __syncthreads();
    if (sel) {
      const int o = run_sel + sel_rank;
      if (o < kTopK) {
        int pos = j;
        if (kCand) pos = (all ? (j >> 3) : __ldg(crow + (j >> 3))) * kCandBlock + (j & 7);
        orow[o] = pos;
      }
    }
    run_eq += eq_tot;
    run_sel += sel_tot;
    if (run_sel >= kTopK) break;
  }
  for (int o = run_sel + tid; o < kTopK; o += kTopThreads) orow[o] = -1;
}

// ------------------------------------------------------------------------------------------- decode (small T)
// VLLM_DS41_INDEXER_FUSED_DECODE: graph-capturable decode path (T <= 16 rows per call, no host decisions).
// * block_max: full scan of a row's visible keys (gathered through the block table), per 8-position block the max
//   score (-inf past the row's end; the block holding the newest visible position pinned to +inf, as
//   select_candidate_blocks); grid (rows, max_n / keys_per_cta), CTAs past a row's end exit at once.
// * topk_sel<K>: radix top-K of a row's first L_row entries (L_row = ceil(end / len_div) when ends is given, else
//   L), ties -> lowest column, -inf never picked, ascending output (slot -> position through cand when given), -1
//   padded; cand_all_max > 0 writes CAND_ALL into column 0 of rows that reach <= cand_all_max blocks. 1024 threads,
//   warp-aggregated histograms (__match_any_sync), one ordered compaction over contiguous per-thread chunks.
__global__ void __launch_bounds__(kThreads, 2)
block_max_kernel(const half* __restrict__ q, int64_t q_s0, int q_s1, const float* __restrict__ w,
                 const int64_t* __restrict__ ends, const half* __restrict__ keys, const int* __restrict__ tok2req,
                 const int* __restrict__ bt, int64_t bt_s0, int storage, float* __restrict__ blk, int64_t blk_ld,
                 int keys_per_cta) {
  __shared__ Smem sm;
  const int t = blockIdx.x;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int64_t end = ends[t];
  const int64_t j_begin = static_cast<int64_t>(blockIdx.y) * keys_per_cta;
  if (j_begin >= end) return;                                   // CTA-uniform, before any barrier
  const int64_t j_end = min(end, j_begin + keys_per_cta);
  const int* btrow = bt + static_cast<int64_t>(__ldg(tok2req + t)) * bt_s0;
  if (tid < kH) sm.w[0][tid] = w[static_cast<int64_t>(t) * kH + tid];
  FragA qa[2][kKT];
  load_q(qa, q + t * q_s0, q_s1);
  float* brow = blk + static_cast<int64_t>(t) * blk_ld;
  const int64_t pinned = (end - 1) / kCandBlock;
  uint4 v[kU];
  auto fetch = [&](int64_t j0) {
#pragma unroll
    for (int u = 0; u < kU; ++u) {
      const int idx = u * kThreads + tid, key = idx >> 4, part = idx & 15;
      const int64_t j = j0 + key;
      if (j < j_end) {
        const int64_t row = static_cast<int64_t>(__ldg(btrow + j / storage)) * storage + j % storage;
        v[u] = __ldg(reinterpret_cast<const uint4*>(keys + row * kD) + part);
      } else {
        v[u] = make_uint4(0, 0, 0, 0);
      }
    }
  };
  fetch(j_begin);
  for (int64_t j0 = j_begin; j0 < j_end; j0 += kBN) {
    __syncthreads();
#pragma unroll
    for (int u = 0; u < kU; ++u) {
      const int idx = u * kThreads + tid, key = idx >> 4, part = idx & 15;
      *reinterpret_cast<uint4*>(sm.k + key * kLdK + part * 8) = v[u];
    }
    __syncthreads();
    if (j0 + kBN < j_end) fetch(j0 + kBN);
    const float s = tile_score(qa, sm.k, warp, sm.s[warp], sm.w[0], lane);
    const int64_t j = j0 + warp * 16 + (lane & 15);
    float m = (lane < 16 && j < j_end) ? s : -INFINITY;
    m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 1));
    m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 2));
    m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 4));
    if ((lane == 0 || lane == 8) && j < j_end) {
      const int64_t b = j / kCandBlock;
      brow[b] = b == pinned ? INFINITY : m;
    }
  }
}

constexpr int kSelThreads = 1024;

template <int K, bool kCand>
__global__ void __launch_bounds__(kSelThreads)
topk_sel_kernel(const float* __restrict__ scores, int64_t ld, int L, const int64_t* __restrict__ ends, int len_div,
                const int* __restrict__ cand, int64_t cand_s0, int* __restrict__ out, int64_t out_s0,
                int cand_all_max) {
  using Scan = cub::BlockScan<int, kSelThreads>;
  __shared__ typename Scan::TempStorage tmp;
  __shared__ int hist[256];
  __shared__ unsigned s_prefix;
  __shared__ int s_k;
  const int t = blockIdx.x, tid = threadIdx.x, lane = tid & 31;
  const float* row = scores + static_cast<int64_t>(t) * ld;
  int* orow = out + static_cast<int64_t>(t) * out_s0;
  int Lrow = L;
  if (ends != nullptr && len_div > 0) {
    const int64_t e = ends[t];
    Lrow = static_cast<int>(min(static_cast<int64_t>(L), (max(e, static_cast<int64_t>(0)) + len_div - 1) / len_div));
  }
  unsigned thr = 0u;
  int need_eq = 0;
  if (Lrow > K) {
    unsigned prefix = 0u, mask = 0u;
    int k = K;
    for (int shift = 24; shift >= 0; shift -= 8) {
      for (int i = tid; i < 256; i += kSelThreads) hist[i] = 0;
      __syncthreads();
      for (int base = 0; base < Lrow; base += kSelThreads) {
        const int j = base + tid;
        const unsigned key = j < Lrow ? order_key(row[j]) : 0u;
        const bool hit = j < Lrow && (key & mask) == prefix;
        const unsigned bin = hit ? ((key >> shift) & 255u) : 256u;
        const unsigned peers = __match_any_sync(0xffffffffu, bin);
        if (hit && lane == __ffs(peers) - 1) atomicAdd(&hist[bin], __popc(peers));
      }
      __syncthreads();
      if (tid == 0) {
        int cum = 0, b = 255;
        for (; b > 0; --b) {
          if (cum + hist[b] >= k) break;
          cum += hist[b];
        }
        s_prefix = prefix | (static_cast<unsigned>(b) << shift);
        s_k = k - cum;
      }
      __syncthreads();
      prefix = s_prefix;
      k = s_k;
      mask |= 0xFFu << shift;
      __syncthreads();
    }
    thr = prefix;
    need_eq = thr == 0u ? 0 : k;
  }
  // ordered compaction over contiguous per-thread chunks (columns ascending = positions ascending)
  const int chunk = (Lrow + kSelThreads - 1) / kSelThreads;
  const int a = min(Lrow, tid * chunk), b = min(Lrow, a + chunk);
  int ngt = 0, neq = 0;
  for (int j = a; j < b; ++j) {
    const unsigned key = order_key(row[j]);
    ngt += key > thr;
    neq += thr != 0u && key == thr;
  }
  int eq_before, eq_total;
  Scan(tmp).ExclusiveSum(neq, eq_before, eq_total);
  __syncthreads();
  const int my_eq = max(0, min(neq, need_eq - eq_before));
  int sel_before, sel_total;
  Scan(tmp).ExclusiveSum(ngt + my_eq, sel_before, sel_total);
  __syncthreads();
  const int* crow = kCand ? cand + t * cand_s0 : nullptr;
  const bool all = kCand && __ldg(crow) == kCandAll;
  int o = sel_before, eq_used = 0;
  for (int j = a; j < b && o < K; ++j) {
    const unsigned key = order_key(row[j]);
    bool sel = key > thr;
    if (!sel && thr != 0u && key == thr && eq_used < my_eq) {
      sel = true;
      ++eq_used;
    }
    if (sel) {
      int pos = j;
      if (kCand) pos = (all ? (j >> 3) : __ldg(crow + (j >> 3))) * kCandBlock + (j & 7);
      orow[o++] = pos;
    }
  }
  const int total = min(K, sel_total);
  for (int i = total + tid; i < K; i += kSelThreads) orow[i] = -1;
  if (cand_all_max > 0 && ends != nullptr) {
    const int64_t reach = (max(ends[t], static_cast<int64_t>(0)) + kCandBlock - 1) / kCandBlock;
    if (reach <= cand_all_max) {
      __syncthreads();
      if (tid == 0) orow[0] = kCandAll;
    }
  }
}

// ------------------------------------------------------------------------------------------- host
static bool aligned(const at::Tensor& x, int bytes) { return reinterpret_cast<uintptr_t>(x.data_ptr()) % bytes == 0; }

static void check_q_w(const at::Tensor& q, const at::Tensor& w) {
  TORCH_CHECK(q.is_cuda() && q.scalar_type() == at::kHalf && q.dim() == 3 && q.size(1) == kH && q.size(2) == kD,
              "q must be fp16 [Q, 32, 128]");
  TORCH_CHECK(q.stride(2) == 1 && q.stride(1) % 16 == 0 && q.stride(0) % 16 == 0 && aligned(q, 32),
              "q: unit inner stride, 32-byte aligned rows");
  TORCH_CHECK(q.stride(1) <= INT32_MAX, "q head stride too large");
  TORCH_CHECK(w.is_cuda() && w.scalar_type() == at::kFloat && w.is_contiguous() && w.dim() == 2 &&
                  w.size(0) == q.size(0) && w.size(1) == kH,
              "w must be contiguous fp32 [Q, 32]");
}

// scores [Q, 16384] fp32 (contiguous) <- candidate slots of each row
void cand_scores(at::Tensor q, at::Tensor w, at::Tensor cand, at::Tensor ends, at::Tensor keys, at::Tensor tok2req,
                 at::Tensor bt, int64_t storage, at::Tensor scores, int64_t slots_per_cta) {
  check_q_w(q, w);
  const int64_t Q = q.size(0);
  TORCH_CHECK(cand.is_cuda() && cand.scalar_type() == at::kInt && cand.dim() == 2 && cand.size(0) == Q &&
                  cand.size(1) == kCandSlots / kCandBlock && cand.stride(1) == 1,
              "cand must be int32 [Q, 2048] with unit inner stride");
  TORCH_CHECK(ends.is_cuda() && ends.scalar_type() == at::kLong && ends.is_contiguous() && ends.numel() == Q,
              "ends must be contiguous int64 [Q]");
  TORCH_CHECK(keys.is_cuda() && keys.scalar_type() == at::kHalf && keys.dim() == 2 && keys.size(1) == kD &&
                  keys.is_contiguous() && aligned(keys, 16),
              "keys must be contiguous fp16 [rows, 128]");
  TORCH_CHECK(tok2req.is_cuda() && tok2req.scalar_type() == at::kInt && tok2req.is_contiguous() &&
                  tok2req.numel() == Q,
              "tok2req must be contiguous int32 [Q]");
  TORCH_CHECK(bt.is_cuda() && bt.scalar_type() == at::kInt && bt.dim() == 2 && bt.stride(1) == 1,
              "block table must be int32 [R, W] with unit inner stride");
  TORCH_CHECK(storage > 0, "storage block size must be > 0");
  TORCH_CHECK(scores.is_cuda() && scores.scalar_type() == at::kFloat && scores.is_contiguous() &&
                  scores.dim() == 2 && scores.size(0) == Q && scores.size(1) == kCandSlots,
              "scores must be contiguous fp32 [Q, 16384]");
  TORCH_CHECK(slots_per_cta > 0 && slots_per_cta % kBN == 0 && kCandSlots % slots_per_cta == 0,
              "slots_per_cta must divide 16384 in multiples of 64");
  if (Q == 0) return;
  const at::cuda::OptionalCUDAGuard guard(q.device());
  const dim3 grid(static_cast<unsigned>(Q), static_cast<unsigned>(kCandSlots / slots_per_cta));
  cand_scores_kernel<<<grid, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const half*>(q.data_ptr<at::Half>()), q.stride(0), static_cast<int>(q.stride(1)),
      w.data_ptr<float>(), cand.data_ptr<int>(), cand.stride(0), ends.data_ptr<int64_t>(),
      reinterpret_cast<const half*>(keys.data_ptr<at::Half>()), tok2req.data_ptr<int>(), bt.data_ptr<int>(),
      bt.stride(0), static_cast<int>(storage), scores.data_ptr<float>(), static_cast<int>(slots_per_cta));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// out[:, :n] fp32 (row stride ld) <- full scan over keys [n, 128]
void dense_scores(at::Tensor q, at::Tensor w, at::Tensor keys, at::Tensor out, int64_t keys_per_cta) {
  check_q_w(q, w);
  const int64_t Q = q.size(0), n = keys.size(0);
  TORCH_CHECK(keys.is_cuda() && keys.scalar_type() == at::kHalf && keys.dim() == 2 && keys.size(1) == kD &&
                  keys.is_contiguous() && aligned(keys, 16),
              "keys must be contiguous fp16 [n, 128]");
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == at::kFloat && out.dim() == 2 && out.size(0) == Q &&
                  out.size(1) >= n && out.stride(1) == 1,
              "out must be fp32 [Q, >= n] with unit inner stride");
  TORCH_CHECK(keys_per_cta > 0 && keys_per_cta % kBN == 0, "keys_per_cta must be a multiple of 64");
  TORCH_CHECK(n < INT32_MAX, "too many keys");
  if (Q == 0 || n == 0) return;
  const at::cuda::OptionalCUDAGuard guard(q.device());
  const dim3 grid(static_cast<unsigned>((Q + kWarps - 1) / kWarps),
                  static_cast<unsigned>((n + keys_per_cta - 1) / keys_per_cta));
  dense_scores_kernel<<<grid, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const half*>(q.data_ptr<at::Half>()), q.stride(0), static_cast<int>(q.stride(1)),
      w.data_ptr<float>(), reinterpret_cast<const half*>(keys.data_ptr<at::Half>()), static_cast<int>(n),
      static_cast<int>(Q), out.data_ptr<float>(), out.stride(0), static_cast<int>(keys_per_cta));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// out [Q, 512] int32 <- top-512 of scores[:, :L] (cand given: slot -> position through it; else column = position)
void topk512(at::Tensor scores, int64_t L, c10::optional<at::Tensor> cand, at::Tensor out) {
  TORCH_CHECK(scores.is_cuda() && scores.scalar_type() == at::kFloat && scores.dim() == 2 && scores.stride(1) == 1 &&
                  scores.size(1) >= L,
              "scores must be fp32 [Q, >= L] with unit inner stride");
  const int64_t Q = scores.size(0);
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == at::kInt && out.dim() == 2 && out.size(0) == Q &&
                  out.size(1) == kTopK && out.stride(1) == 1,
              "out must be int32 [Q, 512] with unit inner stride");
  TORCH_CHECK(L >= 0 && L < INT32_MAX, "bad L");
  if (Q == 0) return;
  const at::cuda::OptionalCUDAGuard guard(scores.device());
  const auto stream = at::cuda::getCurrentCUDAStream();
  if (cand.has_value() && cand->defined()) {
    const at::Tensor& c = *cand;
    TORCH_CHECK(c.is_cuda() && c.scalar_type() == at::kInt && c.dim() == 2 && c.size(0) == Q &&
                    c.size(1) * kCandBlock >= L && c.stride(1) == 1,
                "cand must be int32 [Q, >= L / 8] with unit inner stride");
    topk512_kernel<true><<<static_cast<unsigned>(Q), kTopThreads, 0, stream>>>(
        scores.data_ptr<float>(), scores.stride(0), static_cast<int>(L), c.data_ptr<int>(), c.stride(0),
        out.data_ptr<int>(), out.stride(0));
  } else {
    topk512_kernel<false><<<static_cast<unsigned>(Q), kTopThreads, 0, stream>>>(
        scores.data_ptr<float>(), scores.stride(0), static_cast<int>(L), nullptr, 0, out.data_ptr<int>(),
        out.stride(0));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// blk[:, :ceil(n_t / 8)] fp32 <- per-block max scores of each row's visible keys (decode, graph-capturable)
void block_max(at::Tensor q, at::Tensor w, at::Tensor ends, at::Tensor keys, at::Tensor tok2req, at::Tensor bt,
               int64_t storage, at::Tensor blk, int64_t max_n, int64_t keys_per_cta) {
  check_q_w(q, w);
  const int64_t Q = q.size(0);
  TORCH_CHECK(ends.is_cuda() && ends.scalar_type() == at::kLong && ends.is_contiguous() && ends.numel() == Q,
              "ends must be contiguous int64 [Q]");
  TORCH_CHECK(keys.is_cuda() && keys.scalar_type() == at::kHalf && keys.dim() == 2 && keys.size(1) == kD &&
                  keys.is_contiguous() && aligned(keys, 16),
              "keys must be contiguous fp16 [rows, 128]");
  TORCH_CHECK(tok2req.is_cuda() && tok2req.scalar_type() == at::kInt && tok2req.is_contiguous() &&
                  tok2req.numel() == Q, "tok2req must be contiguous int32 [Q]");
  TORCH_CHECK(bt.is_cuda() && bt.scalar_type() == at::kInt && bt.dim() == 2 && bt.stride(1) == 1,
              "block table must be int32 [R, W] with unit inner stride");
  TORCH_CHECK(storage > 0 && max_n > 0, "storage / max_n must be > 0");
  TORCH_CHECK(keys_per_cta > 0 && keys_per_cta % kBN == 0, "keys_per_cta must be a multiple of 64");
  TORCH_CHECK(blk.is_cuda() && blk.scalar_type() == at::kFloat && blk.dim() == 2 && blk.size(0) >= Q &&
                  blk.size(1) * kCandBlock >= max_n && blk.stride(1) == 1,
              "blk must be fp32 [>= Q, >= max_n / 8] with unit inner stride");
  if (Q == 0) return;
  const at::cuda::OptionalCUDAGuard guard(q.device());
  const dim3 grid(static_cast<unsigned>(Q), static_cast<unsigned>((max_n + keys_per_cta - 1) / keys_per_cta));
  block_max_kernel<<<grid, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const half*>(q.data_ptr<at::Half>()), q.stride(0), static_cast<int>(q.stride(1)),
      w.data_ptr<float>(), ends.data_ptr<int64_t>(), reinterpret_cast<const half*>(keys.data_ptr<at::Half>()),
      tok2req.data_ptr<int>(), bt.data_ptr<int>(), bt.stride(0), static_cast<int>(storage), blk.data_ptr<float>(),
      blk.stride(0), static_cast<int>(keys_per_cta));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// out [Q, k] int32 <- top-k (k = 512 or 2048) of scores[t, :L_row] (see topk_sel_kernel)
void topk_sel(at::Tensor scores, int64_t L, c10::optional<at::Tensor> ends, int64_t len_div,
              c10::optional<at::Tensor> cand, at::Tensor out, int64_t k, int64_t cand_all_max) {
  TORCH_CHECK(scores.is_cuda() && scores.scalar_type() == at::kFloat && scores.dim() == 2 && scores.stride(1) == 1 &&
                  scores.size(1) >= L, "scores must be fp32 [Q, >= L] with unit inner stride");
  const int64_t Q = out.size(0);
  TORCH_CHECK(scores.size(0) >= Q, "scores has fewer rows than out");
  TORCH_CHECK(k == 512 || k == 2048, "k must be 512 or 2048");
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == at::kInt && out.dim() == 2 && out.size(1) == k &&
                  out.stride(1) == 1, "out must be int32 [Q, k] with unit inner stride");
  TORCH_CHECK(L >= 0 && L < INT32_MAX, "bad L");
  const int64_t* e = nullptr;
  if (ends.has_value() && ends->defined()) {
    TORCH_CHECK(ends->is_cuda() && ends->scalar_type() == at::kLong && ends->is_contiguous() && ends->numel() >= Q,
                "ends must be contiguous int64 [>= Q]");
    e = ends->data_ptr<int64_t>();
  }
  if (Q == 0) return;
  const at::cuda::OptionalCUDAGuard guard(scores.device());
  const auto stream = at::cuda::getCurrentCUDAStream();
  const unsigned grid = static_cast<unsigned>(Q);
  if (cand.has_value() && cand->defined()) {
    const at::Tensor& c = *cand;
    TORCH_CHECK(k == 512 && c.is_cuda() && c.scalar_type() == at::kInt && c.dim() == 2 && c.size(0) >= Q &&
                    c.size(1) * kCandBlock >= L && c.stride(1) == 1,
                "cand must be int32 [>= Q, >= L / 8] with unit inner stride (k = 512)");
    topk_sel_kernel<512, true><<<grid, kSelThreads, 0, stream>>>(
        scores.data_ptr<float>(), scores.stride(0), static_cast<int>(L), e, static_cast<int>(len_div),
        c.data_ptr<int>(), c.stride(0), out.data_ptr<int>(), out.stride(0), static_cast<int>(cand_all_max));
  } else if (k == 512) {
    topk_sel_kernel<512, false><<<grid, kSelThreads, 0, stream>>>(
        scores.data_ptr<float>(), scores.stride(0), static_cast<int>(L), e, static_cast<int>(len_div), nullptr, 0,
        out.data_ptr<int>(), out.stride(0), static_cast<int>(cand_all_max));
  } else {
    topk_sel_kernel<2048, false><<<grid, kSelThreads, 0, stream>>>(
        scores.data_ptr<float>(), scores.stride(0), static_cast<int>(L), e, static_cast<int>(len_div), nullptr, 0,
        out.data_ptr<int>(), out.stride(0), static_cast<int>(cand_all_max));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace ds41_indexer_fused

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cand_scores", &ds41_indexer_fused::cand_scores, "DS-V4.1 indexer: candidate-slot scores (SM70 WMMA)");
  m.def("dense_scores", &ds41_indexer_fused::dense_scores, "DS-V4.1 indexer: full-scan scores (SM70 WMMA)");
  m.def("topk512", &ds41_indexer_fused::topk512, "DS-V4.1 indexer: top-512 ascending by position, ties lowest");
  m.attr("cand_slots") = ds41_indexer_fused::kCandSlots;
  m.attr("keys_per_chunk") = ds41_indexer_fused::kBN;
  m.def("block_max", &ds41_indexer_fused::block_max, "DS-V4.1 indexer decode: per-block max scores (SM70 WMMA)");
  m.def("topk_sel", &ds41_indexer_fused::topk_sel, "DS-V4.1 indexer decode: radix top-512 / top-2048 per row");
  m.attr("version") = 2;
}
