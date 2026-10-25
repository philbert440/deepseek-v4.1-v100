// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// EXL3 (trellis-coded, mcg or mul1 codebook, K = 2/3/4 bits, and K = 5 in the
// sm80 tile order) routed-expert MoE for SM70. Weights are decoded on the fly
// from the trellis and multiplied with mma.m8n8k4 (fp32 accumulate); nothing is
// dequantized to memory.
//
// Expert id -1 marks a slot this launch must skip (an expert owned by another
// rank under expert parallelism, or held in another K bucket): gemv, pre and mid
// blocks of such a slot exit, and post leaves it out of the sum.
//
// One slot = one (token, routed expert) assignment, s = t * topk + j.
//   pre  : X'[s] = had128(x[t] * suh[e])                                fp16
//   [S, k] gemv : Y'[s] = X'[s] @ T[e] fp32 [S, n] mid  : H'[s] =
//   had128(silu(had128(Yg') * svh_g[e]) *
//                         (had128(Yu') * svh_u[e]) * suh_d[e])           fp16
//                         [S, n]
//   post : out[t] = sum_j w[s] * had128(O'[s]) * svh_d[e]              fp32 [T,
//   k]
// W = diag(suh) H T H diag(svh), H = blockwise Sylvester-128 / sqrt(128).
//
// Trellis layout (exllamav3): int16 [E, k/16, n/16, 16K]; each 16x16 tile is a
// tail-biting 256K-bit stream (uint16 pairs half-swapped, i.e. MSB-first when
// read as little-endian u32), position p's state = the 16 bits ending at bit
// (p + 1) K. The tile order maps positions to tile elements:
//   colmajor: p -> (k = p % 16, n = p / 16). A lane reads 8 consecutive k of
//             one column per run: two runs per tile.
//   sm80    : exllamav3's default (m16n8k16 B-fragment order). Column c's 16
//             values are four runs of 4 positions, 8K bits apart; one
//             m8n8k4 per run, so a lane decodes 4 short runs from 4-6 words.
// Measured on V100 SXM2 (per-rank GLM-5.3 expert shapes), sm80 vs colmajor:
// +4-13% at K = 2, +12-20% at K = 3 in decode and none in grouped prefill at
// K = 3. Both are exact.
//
// All launches are shape-static per slot count and read expert ids on the
// device, so the decode path is CUDA-graph capturable. The grouped variant
// (prefill) decodes each expert once per <= 8 slots routed to it; the row-tiled
// variant (below, VLLM_DS41_MOE_EXL3_PREFILL_KERNEL=1) once per 16 / 32 / 64.

#include <torch/all.h>
#include <torch/library.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace {

constexpr int kWarps = 4;      // gemv: warps per block, each a k-slice
constexpr int kPrefetch = 8;   // gemv: k-tiles whose window words are in flight per warp
constexpr int kGlueWarps = 4;  // glue kernels: one warp per 128-chunk
constexpr int kMaxGridY = 65535;
// Trellis tensors reach these ops in strip-major order: [.., n/32, k/16, 2 tiles, 16K words]
// (exl3.exl3_strip_major_), so a warp streams its 32-column strip sequentially along k.

__device__ __forceinline__ void mma884(float (&d)[8], const uint32_t (&a)[2],
                                       const uint32_t (&b)[2]) {
  asm volatile(
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
      "{%0,%1,%2,%3,%4,%5,%6,%7};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]),
        "+f"(d[6]), "+f"(d[7])
      : "r"(a[0]), "r"(a[1]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ half2 mcg2(uint32_t s0, uint32_t s1) {
  uint32_t x0 = s0 * 0xCBAC1FEDu, x1 = s1 * 0xCBAC1FEDu;
  asm("lop3.b32 %0, %0, 0x8fff8fff, 0x3b603b60, 0x6a;" : "+r"(x0));
  asm("lop3.b32 %0, %0, 0x8fff8fff, 0x3b603b60, 0x6a;" : "+r"(x1));
  half2 h0 = *reinterpret_cast<half2*>(&x0);
  half2 h1 = *reinterpret_cast<half2*>(&x1);
  return __hadd2(__lows2half2(h0, h1), __highs2half2(h0, h1));
}

// exllamav3's mul1 codebook (exllamav3_ext/quant/codebook.cuh,
// decode_mul1_product_2): byte sum of state * 0x83DCD12D on top of 0x6400
// (1024.0), then one fp16 fma with 1/147.7 and -10.39.
__device__ __forceinline__ half2 mul1_2(uint32_t s0, uint32_t s1) {
  uint32_t x0 = s0 * 0x83DCD12Du, x1 = s1 * 0x83DCD12Du;
  uint32_t sum0 = __dp4a(x0, 0x01010101u, 0x6400u);
  uint32_t sum1 = __dp4a(x1, 0x01010101u, 0x6400u);
  half2 h = __halves2half2(__ushort_as_half(static_cast<uint16_t>(sum0)),
                           __ushort_as_half(static_cast<uint16_t>(sum1)));
  return __hfma2(h, __half2half2(__ushort_as_half(0x1eee)),
                 __half2half2(__ushort_as_half(0xc931)));
}

constexpr int kCodebookMcg = 0;
constexpr int kCodebookMul1 = 1;

template <int CB>
__device__ __forceinline__ half2 decode2(uint32_t s0, uint32_t s1) {
  if constexpr (CB == kCodebookMul1) {
    return mul1_2(s0, s1);
  } else {
    return mcg2(s0, s1);
  }
}

// N consecutive positions from the window starting at bit o of (wa, wb).
template <int K, int N, int CB>
__device__ __forceinline__ void decode_run(uint32_t wa, uint32_t wb, int o,
                                           uint32_t (&b)[N / 2]) {
  uint64_t v = (static_cast<uint64_t>(wa) << 32) | wb;
  uint32_t st[N];
#pragma unroll
  for (int i = 0; i < N; ++i) {
    st[i] = static_cast<uint32_t>(v >> (48 - o - i * K)) & 0xffffu;
  }
#pragma unroll
  for (int j = 0; j < N / 2; ++j) {
    half2 r = decode2<CB>(st[2 * j], st[2 * j + 1]);
    b[j] = *reinterpret_cast<uint32_t*>(&r);
  }
}

// Lane-constant window geometry for tile t (0/1 within the strip), column c.
// colmajor: runs h = 0, 1 at positions 16c + 8h. sm80: run 0 only (the base);
// runs 1-3 follow at +8K bits each.
template <int K, bool kColMajor>
struct Window {
  static constexpr int kRuns = kColMajor ? 2 : 1;
  int ja[kRuns], jb[kRuns], o[kRuns];
  __device__ __forceinline__ Window(int t, int c) {
    constexpr int kTW = 8 * K, kLB = 256 * K;
#pragma unroll
    for (int r = 0; r < kRuns; ++r) {
      int pos0 = kColMajor ? 16 * c + 8 * r : 32 * (c & 7) + 4 * (c >> 3);
      int s = (((pos0 + 1) * K - 16) % kLB + kLB) % kLB;
      ja[r] = t * kTW + (s >> 5);
      jb[r] = t * kTW + ((s >> 5) + 1) % kTW;
      o[r] = s & 31;
    }
  }
};

// The words of the two-tile block this lane's windows need, loaded directly
// (every lane's words sit in the same 128 B line, so DRAM traffic equals one
// coalesced block read, but each lane can decode as soon as its own loads
// land instead of waiting for a broadcast and shuffles).
template <int K, bool kColMajor>
struct LaneWords {
  static constexpr int kTW = 8 * K;
  static constexpr int kCount = kColMajor ? 4 : ((24 * K) >> 5) + 3;
  uint32_t w[kCount];
  __device__ __forceinline__ void load(const uint32_t* __restrict__ blk,
                                       const Window<K, kColMajor>& win) {
    if constexpr (kColMajor) {
      w[0] = __ldg(blk + win.ja[0]);
      w[1] = __ldg(blk + win.jb[0]);
      w[2] = __ldg(blk + win.ja[1]);
      w[3] = __ldg(blk + win.jb[1]);
    } else {
      const int base = win.ja[0] % kTW, t0 = win.ja[0] - base;
#pragma unroll
      for (int m = 0; m < kCount; ++m) w[m] = __ldg(blk + t0 + (base + m) % kTW);
    }
  }
  __device__ __forceinline__ void zero() {
#pragma unroll
    for (int m = 0; m < kCount; ++m) w[m] = 0;
  }
};

// Strip-major trellis address of k-tile tk for the strip starting at column
// n0: tile pairs of one strip are contiguous along k.
__device__ __forceinline__ const uint32_t* strip_block(
    const uint32_t* __restrict__ tr, int n0, int tk, int ktiles, int kTW) {
  return tr + (static_cast<int64_t>(n0 >> 5) * ktiles + tk) * (2 * kTW);
}

// One 16-row k-tile of this lane's column. xr points at the tile's 16
// activations (row 0 of the m8n8k4 A operand; other rows pass on = false).
template <int K, bool kColMajor, int CB>
__device__ __forceinline__ void tile_mma(float (&d)[8],
                                         const LaneWords<K, kColMajor>& lw,
                                         const Window<K, kColMajor>& win,
                                         const half* __restrict__ xr, bool on) {
  if constexpr (kColMajor) {
    uint32_t bb[2][4];
    decode_run<K, 8, CB>(lw.w[0], lw.w[1], win.o[0], bb[0]);
    decode_run<K, 8, CB>(lw.w[2], lw.w[3], win.o[1], bb[1]);
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      uint32_t b0[2] = {bb[h][0], bb[h][1]}, b1[2] = {bb[h][2], bb[h][3]};
      uint4 xa = on ? *reinterpret_cast<const uint4*>(xr + 8 * h)
                    : make_uint4(0, 0, 0, 0);
      uint32_t a0[2] = {xa.x, xa.y}, a1[2] = {xa.z, xa.w};
      mma884(d, a0, b0);
      mma884(d, a1, b1);
    }
  } else {
    constexpr int kNW = ((24 * K) >> 5) + 3;
    uint4 lo =
        on ? *reinterpret_cast<const uint4*>(xr) : make_uint4(0, 0, 0, 0);
    uint4 hi =
        on ? *reinterpret_cast<const uint4*>(xr + 8) : make_uint4(0, 0, 0, 0);
    const uint32_t xw[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
    const int o0 = win.o[0] & 31;
    static_assert(LaneWords<K, kColMajor>::kCount == kNW);
    const uint32_t* w = lw.w;
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      // Run u holds rows (2u, 2u+1, 2u+8, 2u+9): A = activation pairs u, u+4.
      constexpr int kStep = 8 * K;
      const int ml = (kStep * u) >> 5, ol = (kStep * u) & 31;
      const int ou = o0 + ol;
      const bool carry = ou >= 32;
      uint32_t b[2];
      decode_run<K, 4, CB>(carry ? w[ml + 1] : w[ml],
                           carry ? w[ml + 2] : w[ml + 1], ou & 31, b);
      uint32_t a[2] = {xw[u], xw[u + 4]};
      mma884(d, a, b);
    }
  }
}

// grid (n / 32, S or 2S), block kWarps warps. Slot s multiplies expert eid[s].
// M = 1: row 0 of the m8n8k4 tile carries the activation.
template <int K, bool kColMajor, int CB>
__global__ void __launch_bounds__(kWarps * 32)
    exl3_gemv_kernel(const uint32_t* __restrict__ trellis,
                     const int* __restrict__ eid, const half* __restrict__ x,
                     float* __restrict__ y, int k, int n, int ktiles_per_warp,
                     int64_t expert_words, int S,
                     const uint32_t* __restrict__ trellis2,
                     const half* __restrict__ x2, float* __restrict__ y2) {
  int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  int s = blockIdx.y;
  if (s >= S) {  // second projection of a fused gate + up launch
    s -= S;
    trellis = trellis2;
    x = x2;
    y = y2;
  }
  if (eid[s] < 0) return;  // not this launch's expert (whole block)
  int n0 = blockIdx.x * 32;
  int q = (lane >> 2) & 3;
  int nw = 8 * q + (lane >> 4) * 4 + lane % 4;
  int t = nw >> 4, c = nw & 15;
  int row = (lane >> 4) * 4 + lane % 4;
  int tk0 = warp * ktiles_per_warp;
  const uint32_t* tr = trellis + static_cast<int64_t>(eid[s]) * expert_words;
  const half* xr = x + static_cast<int64_t>(s) * k;
  constexpr int kTW = 8 * K;
  Window<K, kColMajor> win(t, c);

  float d[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  const int ktiles = ktiles_per_warp * kWarps;
  for (int it0 = 0; it0 < ktiles_per_warp; it0 += kPrefetch) {
    LaneWords<K, kColMajor> lw[kPrefetch];
#pragma unroll
    for (int p = 0; p < kPrefetch; ++p) {
      if (it0 + p < ktiles_per_warp) {
        lw[p].load(strip_block(tr, n0, tk0 + it0 + p, ktiles, kTW), win);
      } else {
        lw[p].zero();
      }
    }
#pragma unroll
    for (int p = 0; p < kPrefetch; ++p) {
      if (it0 + p >= ktiles_per_warp) break;
      int tk = tk0 + it0 + p;
      tile_mma<K, kColMajor, CB>(d, lw[p], win, xr + tk * 16, row == 0);
    }
  }

  // D fragment: lane L, reg i -> row (L&1) + ((i>>1)&1)*2 + (L>=16 ? 4 : 0),
  // col ((i>>2)&1)*4 + (L&2) + (i&1)
  __shared__ float red[kWarps][32];
  if (lane < 16 && !(lane & 1)) {
    red[warp][8 * q + (lane & 2)] = d[0];
    red[warp][8 * q + (lane & 2) + 1] = d[1];
    red[warp][8 * q + 4 + (lane & 2)] = d[4];
    red[warp][8 * q + 4 + (lane & 2) + 1] = d[5];
  }
  __syncthreads();
  if (threadIdx.x < 32) {
    float acc = 0.f;
#pragma unroll
    for (int w = 0; w < kWarps; ++w) acc += red[w][threadIdx.x];
    y[static_cast<int64_t>(s) * n + n0 + threadIdx.x] = acc;
  }
}

// Grouped variant (prefill): a group is up to 8 slots routed to the same
// expert; its tiles are decoded once and multiplied against all 8 rows.
// rows[g * 8 + m] = slot or -1. Launches are sized for the worst case; groups
// with rows[g * 8] < 0 exit immediately.
template <int K, bool kColMajor, int CB>
__global__ void __launch_bounds__(kWarps * 32) exl3_gemv_grouped_kernel(
    const uint32_t* __restrict__ trellis, const int* __restrict__ gexp,
    const int* __restrict__ rows, const half* __restrict__ x,
    float* __restrict__ y, int k, int n, int ktiles_per_warp,
    int64_t expert_words, int G, const uint32_t* __restrict__ trellis2,
    const half* __restrict__ x2, float* __restrict__ y2) {
  int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  int g = blockIdx.y;
  if (g >= G) {
    g -= G;
    trellis = trellis2;
    x = x2;
    y = y2;
  }
  const int* gr = rows + static_cast<int64_t>(g) * 8;
  if (gr[0] < 0 || gexp[g] < 0) return;
  int n0 = blockIdx.x * 32;
  int q = (lane >> 2) & 3;
  int nw = 8 * q + (lane >> 4) * 4 + lane % 4;
  int t = nw >> 4, c = nw & 15;
  int row = (lane >> 4) * 4 + lane % 4;  // A-operand row held by this lane
  int slot = gr[row];
  int tk0 = warp * ktiles_per_warp;
  const uint32_t* tr = trellis + static_cast<int64_t>(gexp[g]) * expert_words;
  const half* xr = x + static_cast<int64_t>(slot < 0 ? 0 : slot) * k;
  constexpr int kTW = 8 * K;
  Window<K, kColMajor> win(t, c);

  float d[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  const int ktiles = ktiles_per_warp * kWarps;
  for (int it0 = 0; it0 < ktiles_per_warp; it0 += kPrefetch) {
    LaneWords<K, kColMajor> lw[kPrefetch];
#pragma unroll
    for (int p = 0; p < kPrefetch; ++p) {
      if (it0 + p < ktiles_per_warp) {
        lw[p].load(strip_block(tr, n0, tk0 + it0 + p, ktiles, kTW), win);
      } else {
        lw[p].zero();
      }
    }
#pragma unroll
    for (int p = 0; p < kPrefetch; ++p) {
      if (it0 + p >= ktiles_per_warp) break;
      int tk = tk0 + it0 + p;
      tile_mma<K, kColMajor, CB>(d, lw[p], win, xr + tk * 16, slot >= 0);
    }
  }

  __shared__ float red[kWarps][8][32];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    red[warp][(lane & 1) + ((i >> 1) & 1) * 2 + (lane >= 16 ? 4 : 0)]
       [8 * q + ((i >> 2) & 1) * 4 + (lane & 2) + (i & 1)] = d[i];
  }
  __syncthreads();
  for (int e = threadIdx.x; e < 256; e += kWarps * 32) {
    int m = e >> 5, nn = e & 31;
    int so = gr[m];
    if (so < 0) continue;
    float acc = 0.f;
#pragma unroll
    for (int w = 0; w < kWarps; ++w) acc += red[w][m][nn];
    y[static_cast<int64_t>(so) * n + n0 + nn] = acc;
  }
}

// ---- mixed-K launches: per-expert bit widths in one launch ----
// The experts of a launch live in one flat word buffer: expert e's strip-major
// trellis starts at word offs[e] and has kbits[e] bits per weight. Every block
// serves one slot (or group), so its K is block-uniform and the switch below
// costs one branch; gate + up of an expert share offsets and K.

// One block's share of one slot: columns n0 .. n0 + 31 of y = x @ T.
// With split-K the block covers k-tiles [tk_base, tk_base + kWarps * ktiles_per_warp) of the ktiles_total tiles
// and writes its partial sum; the consumer (mid / post) adds the ksplit partials in fixed order.
template <int K, bool kColMajor, int CB>
__device__ __forceinline__ void gemv_slot_body(const uint32_t* __restrict__ tr,
                                               const half* __restrict__ xr,
                                               float* __restrict__ yr, int n0,
                                               int ktiles_per_warp, int tk_base,
                                               int ktiles_total) {
  int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  int q = (lane >> 2) & 3;
  int nw = 8 * q + (lane >> 4) * 4 + lane % 4;
  int t = nw >> 4, c = nw & 15;
  int row = (lane >> 4) * 4 + lane % 4;
  int tk0 = tk_base + warp * ktiles_per_warp;
  constexpr int kTW = 8 * K;
  Window<K, kColMajor> win(t, c);
  float d[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  const int ktiles = ktiles_total;
  for (int it0 = 0; it0 < ktiles_per_warp; it0 += kPrefetch) {
    LaneWords<K, kColMajor> lw[kPrefetch];
#pragma unroll
    for (int p = 0; p < kPrefetch; ++p) {
      if (it0 + p < ktiles_per_warp) {
        lw[p].load(strip_block(tr, n0, tk0 + it0 + p, ktiles, kTW), win);
      } else {
        lw[p].zero();
      }
    }
#pragma unroll
    for (int p = 0; p < kPrefetch; ++p) {
      if (it0 + p >= ktiles_per_warp) break;
      int tk = tk0 + it0 + p;
      tile_mma<K, kColMajor, CB>(d, lw[p], win, xr + tk * 16, row == 0);
    }
  }
  __shared__ float red[kWarps][32];
  if (lane < 16 && !(lane & 1)) {
    red[warp][8 * q + (lane & 2)] = d[0];
    red[warp][8 * q + (lane & 2) + 1] = d[1];
    red[warp][8 * q + 4 + (lane & 2)] = d[4];
    red[warp][8 * q + 4 + (lane & 2) + 1] = d[5];
  }
  __syncthreads();
  if (threadIdx.x < 32) {
    float acc = 0.f;
#pragma unroll
    for (int w = 0; w < kWarps; ++w) acc += red[w][threadIdx.x];
    yr[n0 + threadIdx.x] = acc;
  }
}

// One block's share of one group (<= 8 slots of one expert, rows gr[0..7]).
template <int K, bool kColMajor, int CB>
__device__ __forceinline__ void gemv_group_body(const uint32_t* __restrict__ tr,
                                                const int* __restrict__ gr,
                                                const half* __restrict__ x,
                                                float* __restrict__ y, int k,
                                                int n, int n0,
                                                int ktiles_per_warp) {
  int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  int q = (lane >> 2) & 3;
  int nw = 8 * q + (lane >> 4) * 4 + lane % 4;
  int t = nw >> 4, c = nw & 15;
  int row = (lane >> 4) * 4 + lane % 4;
  int slot = gr[row];
  int tk0 = warp * ktiles_per_warp;
  const half* xr = x + static_cast<int64_t>(slot < 0 ? 0 : slot) * k;
  constexpr int kTW = 8 * K;
  Window<K, kColMajor> win(t, c);
  float d[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  const int ktiles = ktiles_per_warp * kWarps;
  for (int it0 = 0; it0 < ktiles_per_warp; it0 += kPrefetch) {
    LaneWords<K, kColMajor> lw[kPrefetch];
#pragma unroll
    for (int p = 0; p < kPrefetch; ++p) {
      if (it0 + p < ktiles_per_warp) {
        lw[p].load(strip_block(tr, n0, tk0 + it0 + p, ktiles, kTW), win);
      } else {
        lw[p].zero();
      }
    }
#pragma unroll
    for (int p = 0; p < kPrefetch; ++p) {
      if (it0 + p >= ktiles_per_warp) break;
      int tk = tk0 + it0 + p;
      tile_mma<K, kColMajor, CB>(d, lw[p], win, xr + tk * 16, slot >= 0);
    }
  }
  __shared__ float red[kWarps][8][32];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    red[warp][(lane & 1) + ((i >> 1) & 1) * 2 + (lane >= 16 ? 4 : 0)]
       [8 * q + ((i >> 2) & 1) * 4 + (lane & 2) + (i & 1)] = d[i];
  }
  __syncthreads();
  for (int e = threadIdx.x; e < 256; e += kWarps * 32) {
    int m = e >> 5, nn = e & 31;
    int so = gr[m];
    if (so < 0) continue;
    float acc = 0.f;
#pragma unroll
    for (int w = 0; w < kWarps; ++w) acc += red[w][m][nn];
    y[static_cast<int64_t>(so) * n + n0 + nn] = acc;
  }
}

#define EXL3_K_SWITCH(KB, CALL)          \
  switch (KB) {                          \
    case 2: CALL(2); break;              \
    case 3: CALL(3); break;              \
    case 4: CALL(4); break;              \
    case 5:                              \
      if constexpr (!kColMajor) CALL(5); \
      break;                             \
    default: break;                      \
  }

// grid (n / 32, S or 2S): slot s multiplies expert eid[s] (-1: skip).
template <bool kColMajor, int CB>
__global__ void __launch_bounds__(kWarps * 32) exl3_gemv_mixed_kernel(
    const uint32_t* __restrict__ trellis, const int64_t* __restrict__ offs,
    const int* __restrict__ kbits, const int* __restrict__ eid,
    const half* __restrict__ x, float* __restrict__ y, int k, int n,
    int ktiles_per_warp, int S, const uint32_t* __restrict__ trellis2,
    const half* __restrict__ x2, float* __restrict__ y2, int64_t split_stride) {
  int s = blockIdx.y;
  if (s >= S) {
    s -= S;
    trellis = trellis2;
    x = x2;
    y = y2;
  }
  const int e = eid[s];
  if (e < 0) return;
  const int z = blockIdx.z;  // k-split: partial z lands at rows [z * split_rows, ...)
  const uint32_t* tr = trellis + offs[e];
  const half* xr = x + static_cast<int64_t>(s) * k;
  float* yr = y + z * split_stride + static_cast<int64_t>(s) * n;
  const int n0 = blockIdx.x * 32;
  const int ktiles_total = k / 16;
  const int tk_base = z * ktiles_per_warp * kWarps;
#define EXL3_SLOT(KK)                                                        \
  gemv_slot_body<KK, kColMajor, CB>(tr, xr, yr, n0, ktiles_per_warp, tk_base, \
                                    ktiles_total)
  EXL3_K_SWITCH(kbits[e], EXL3_SLOT)
#undef EXL3_SLOT
}

// grid (n / 32, G or 2G): group g = slots rows[g * 8 ..] of expert gexp[g].
template <bool kColMajor, int CB>
__global__ void __launch_bounds__(kWarps * 32) exl3_gemv_mixed_grouped_kernel(
    const uint32_t* __restrict__ trellis, const int64_t* __restrict__ offs,
    const int* __restrict__ kbits, const int* __restrict__ gexp,
    const int* __restrict__ rows, const half* __restrict__ x,
    float* __restrict__ y, int k, int n, int ktiles_per_warp, int G,
    const uint32_t* __restrict__ trellis2, const half* __restrict__ x2,
    float* __restrict__ y2) {
  int g = blockIdx.y;
  if (g >= G) {
    g -= G;
    trellis = trellis2;
    x = x2;
    y = y2;
  }
  const int* gr = rows + static_cast<int64_t>(g) * 8;
  const int e = gexp[g];
  if (gr[0] < 0 || e < 0) return;
  const uint32_t* tr = trellis + offs[e];
  const int n0 = blockIdx.x * 32;
#define EXL3_GROUP(KK) \
  gemv_group_body<KK, kColMajor, CB>(tr, gr, x, y, k, n, n0, ktiles_per_warp)
  EXL3_K_SWITCH(kbits[e], EXL3_GROUP)
#undef EXL3_GROUP
}
#undef EXL3_K_SWITCH

// 128-point Walsh-Hadamard (Sylvester order) in registers: one warp per
// 128-chunk, lane l holds elements 4l .. 4l+3.
__device__ __forceinline__ void fwht128(float (&v)[4]) {
  int lane = threadIdx.x & 31;
  float a = v[0] + v[1], b = v[0] - v[1], c = v[2] + v[3], d = v[2] - v[3];
  v[0] = a + c;
  v[1] = b + d;
  v[2] = a - c;
  v[3] = b - d;
#pragma unroll
  for (int dl = 1; dl < 32; dl <<= 1) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      float o = __shfl_xor_sync(0xffffffffu, v[i], dl);
      v[i] = (lane & dl) ? o - v[i] : v[i] + o;
    }
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) v[i] *= 0.08838834764831845f;  // 1 / sqrt(128)
}

__device__ __forceinline__ void load4h(const half* p, float (&v)[4]) {
  uint2 r = *reinterpret_cast<const uint2*>(p);
  float2 f0 = __half22float2(*reinterpret_cast<half2*>(&r.x));
  float2 f1 = __half22float2(*reinterpret_cast<half2*>(&r.y));
  v[0] = f0.x;
  v[1] = f0.y;
  v[2] = f1.x;
  v[3] = f1.y;
}

__device__ __forceinline__ void store4h(half* p, const float (&v)[4]) {
  half2 h0 = __floats2half2_rn(v[0], v[1]), h1 = __floats2half2_rn(v[2], v[3]);
  uint2 r;
  r.x = *reinterpret_cast<uint32_t*>(&h0);
  r.y = *reinterpret_cast<uint32_t*>(&h1);
  *reinterpret_cast<uint2*>(p) = r;
}

// Element offset of this lane's 4 values; each warp owns one 128-chunk. Widths
// are multiples of 128, so `offset >= width` is warp-uniform and a warp past the
// end exits whole (no shuffle with missing lanes).
__device__ __forceinline__ int glue_offset() {
  return (blockIdx.x * kGlueWarps + (threadIdx.x >> 5)) * 128 +
         (threadIdx.x & 31) * 4;
}

__host__ __device__ constexpr int glue_blocks(int width) {
  return (width / 128 + kGlueWarps - 1) / kGlueWarps;
}

__global__ void exl3_pre_kernel(const half* __restrict__ x,
                                const int* __restrict__ eid,
                                const half* __restrict__ suh,
                                half* __restrict__ out, int k, int topk, int S,
                                const half* __restrict__ suh2,
                                half* __restrict__ out2) {
  int s = blockIdx.y;
  if (s >= S) {
    s -= S;
    suh = suh2;
    out = out2;
  }
  if (eid[s] < 0) return;
  int o = glue_offset();
  if (o >= k) return;
  float v[4], g[4];
  load4h(x + static_cast<int64_t>(s / topk) * k + o, v);
  load4h(suh + static_cast<int64_t>(eid[s]) * k + o, g);
#pragma unroll
  for (int i = 0; i < 4; ++i) v[i] *= g[i];
  fwht128(v);
  store4h(out + static_cast<int64_t>(s) * k + o, v);
}

__global__ void exl3_mid_kernel(const float* __restrict__ yg,
                                const float* __restrict__ yu,
                                const int* __restrict__ eid,
                                const half* __restrict__ svh_g,
                                const half* __restrict__ svh_u,
                                const half* __restrict__ suh_d,
                                half* __restrict__ out, int n, float limit,
                                int ksplit, int split_rows) {
  int s = blockIdx.y;
  if (eid[s] < 0) return;
  int o = glue_offset();
  if (o >= n) return;
  int64_t e = static_cast<int64_t>(eid[s]) * n + o;
  float g[4] = {0.f, 0.f, 0.f, 0.f}, u[4] = {0.f, 0.f, 0.f, 0.f}, sg[4], su[4], sd[4];
  for (int z = 0; z < ksplit; ++z) {  // split-K partials, fixed order
    const int64_t r = (static_cast<int64_t>(z) * split_rows + s) * n + o;
    float4 a = *reinterpret_cast<const float4*>(yg + r);
    float4 b = *reinterpret_cast<const float4*>(yu + r);
    g[0] += a.x;
    g[1] += a.y;
    g[2] += a.z;
    g[3] += a.w;
    u[0] += b.x;
    u[1] += b.y;
    u[2] += b.z;
    u[3] += b.w;
  }
  fwht128(g);
  fwht128(u);
  load4h(svh_g + e, sg);
  load4h(svh_u + e, su);
  load4h(suh_d + e, sd);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    float gg = g[i] * sg[i], uu = u[i] * su[i];
    if (limit > 0.f) {  // clamped SwiGLU (DeepSeek-V4.1: gate <= limit, |up| <= limit)
      gg = fminf(gg, limit);
      uu = fminf(fmaxf(uu, -limit), limit);
    }
    g[i] = gg / (1.f + __expf(-gg)) * uu * sd[i];
  }
  fwht128(g);
  store4h(out + static_cast<int64_t>(s) * n + o, g);
}

__global__ void exl3_post_kernel(const float* __restrict__ yd,
                                 const int* __restrict__ eid,
                                 const float* __restrict__ w,
                                 const half* __restrict__ svh_d,
                                 float* __restrict__ out, int n, int topk,
                                 int ksplit, int split_rows) {
  int t = blockIdx.y;
  int o = glue_offset();
  if (o >= n) return;
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  for (int j = 0; j < topk; ++j) {
    int s = t * topk + j;
    if (eid[s] < 0) continue;  // warp-uniform: every lane reads the same slot
    float v[4] = {0.f, 0.f, 0.f, 0.f}, sv[4];
    for (int z = 0; z < ksplit; ++z) {  // split-K partials, fixed order
      float4 a = *reinterpret_cast<const float4*>(
          yd + (static_cast<int64_t>(z) * split_rows + s) * n + o);
      v[0] += a.x;
      v[1] += a.y;
      v[2] += a.z;
      v[3] += a.w;
    }
    fwht128(v);
    load4h(svh_d + static_cast<int64_t>(eid[s]) * n + o, sv);
    float ww = w[s];
#pragma unroll
    for (int i = 0; i < 4; ++i) acc[i] += ww * v[i] * sv[i];
  }
  *reinterpret_cast<float4*>(out + static_cast<int64_t>(t) * n + o) =
      make_float4(acc[0], acc[1], acc[2], acc[3]);
}

void check_cuda(const torch::Tensor& t, const char* name, at::ScalarType dtype,
                int dim, const torch::Tensor& ref) {
  TORCH_CHECK(t.is_cuda() && t.device() == ref.device(), name,
              " must be on the same CUDA device as the input");
  TORCH_CHECK(t.scalar_type() == dtype, name, " has the wrong dtype");
  TORCH_CHECK(t.dim() == dim, name, " must have ", dim, " dimensions");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

// trellis [E, k/16, n/16, 16K] -> (k, n, K)
std::tuple<int, int, int> trellis_dims(const torch::Tensor& trellis,
                                       const torch::Tensor& ref) {
  check_cuda(trellis, "trellis", at::kShort, 4, ref);
  TORCH_CHECK(trellis.size(3) % 16 == 0, "trellis: last dim must be 16 * K");
  int K = static_cast<int>(trellis.size(3) / 16);
  TORCH_CHECK(K >= 2 && K <= 5, "EXL3 MoE: K must be 2, 3, 4 or 5");
  int k = static_cast<int>(trellis.size(1) * 16);
  int n = static_cast<int>(trellis.size(2) * 16);
  TORCH_CHECK((k / 16) % kWarps == 0 && n % 32 == 0,
              "EXL3 MoE: k must be a multiple of 64 and n of 32");
  return {k, n, K};
}

template <bool kColMajor, int CB>
void launch_gemv(bool grouped, int K, dim3 grid, cudaStream_t stream,
                 const uint32_t* tr, const int* ids, const int* rows,
                 const half* x, float* y, int k, int n, int64_t ew, int count,
                 const uint32_t* tr2, const half* x2, float* y2) {
  const int kt = k / 16 / kWarps;
#define EXL3_LAUNCH(KK)                                                  \
  if (grouped) {                                                         \
    exl3_gemv_grouped_kernel<KK, kColMajor, CB>                          \
        <<<grid, kWarps * 32, 0, stream>>>(tr, ids, rows, x, y, k, n, kt, \
                                           ew, count, tr2, x2, y2);      \
  } else {                                                               \
    exl3_gemv_kernel<KK, kColMajor, CB><<<grid, kWarps * 32, 0, stream>>>( \
        tr, ids, x, y, k, n, kt, ew, count, tr2, x2, y2);                \
  }
  if (K == 2) {
    EXL3_LAUNCH(2)
  } else if (K == 3) {
    EXL3_LAUNCH(3)
  } else if (K == 4) {
    EXL3_LAUNCH(4)
  } else {
    // K = 5: four positions of a run span 16 + 3 * 5 = 31 bits, which still fit
    // the 64-bit window at any offset < 32; colmajor's 8-position runs do not.
    if constexpr (!kColMajor) {
      EXL3_LAUNCH(5)
    }
  }
#undef EXL3_LAUNCH
}

template <bool kColMajor>
void launch_gemv_cb(int codebook, bool grouped, int K, dim3 grid,
                    cudaStream_t stream, const uint32_t* tr, const int* ids,
                    const int* rows, const half* x, float* y, int k, int n,
                    int64_t ew, int count, const uint32_t* tr2, const half* x2,
                    float* y2) {
  if (codebook == kCodebookMul1) {
    launch_gemv<kColMajor, kCodebookMul1>(grouped, K, grid, stream, tr, ids,
                                          rows, x, y, k, n, ew, count, tr2, x2,
                                          y2);
  } else {
    launch_gemv<kColMajor, kCodebookMcg>(grouped, K, grid, stream, tr, ids,
                                         rows, x, y, k, n, ew, count, tr2, x2,
                                         y2);
  }
}

// ---- prefill: row-tiled launches (VLLM_DS41_MOE_EXL3_PREFILL_KERNEL=1) ----
// The grouped kernel above decodes an expert's trellis once per <= 8 slots, so
// a hot expert with 100 routes is decoded 13 times and the decode (ALU) is the
// cost. Here the routes of this rank are compacted and sorted by expert (rows
// r = 0 .. R-1, plan kernel), and one block decodes each 16-row k-tile of its
// columns once per row tile of up to MT = 16 / 32 / 64 rows and runs the
// m8n8k4 MMA over every 8-row group of the tile from the same registers.
//   block   : kWarps warps, warp w = one 32-column strip (columns n0 .. n0+31),
//             each warp walks all of k (no split-K, no shared memory); the four
//             warps of a block read the same A rows (L1 reuse).
//   lane    : decodes 16 weights per k-tile (four sm80 runs) into 8 B regs,
//             then per 8-row group loads its A row (2 x 16 B) and issues four
//             mma.m8n8k4 into d[g][8] (fp32). MT = 64: 64 accumulators.
// Rows are independent in the MMA (row i of D only reads row i of A), so the
// row order inside a tile does not change any result; the k summation order
// is fixed per column (k-tile order, 4 runs per tile) but differs from the
// 8-slot kernel (one fp32 chain instead of 4 warps' partial sums).
// sm80 tile order only (every DeepSeek-V4.1 EXL3 tensor is sm80).

constexpr int kTileCols = kWarps * 32;  // columns per block
#ifndef EXL3_TILED_P16
#define EXL3_TILED_P16 2
#endif
#ifndef EXL3_TILED_P32
#define EXL3_TILED_P32 2
#endif
#ifndef EXL3_TILED_P64
#define EXL3_TILED_P64 1
#endif
#ifndef EXL3_TILED_KW_P16
#define EXL3_TILED_KW_P16 2
#endif
#ifndef EXL3_TILED_KW_P32
#define EXL3_TILED_KW_P32 2
#endif
#ifndef EXL3_TILED_KW_P64
#define EXL3_TILED_KW_P64 1
#endif
#ifndef EXL3_TILED_MINB16
#define EXL3_TILED_MINB16 4
#endif
#ifndef EXL3_TILED_MINB32
#define EXL3_TILED_MINB32 3
#endif
#ifndef EXL3_TILED_MINB64
#define EXL3_TILED_MINB64 3
#endif
constexpr int kPlanThreads = 1024;
constexpr int kTiers = 3;               // row tiles of 16, 32 and 64 rows
constexpr int kInfoHead = 8;            // info: R, tiles per tier [3], dense count, pad[3]
constexpr int kReconTiles = 8;          // reconstruct: k-tiles per warp

// The four runs of one sm80 k-tile of this lane's column as m8n8k4 B
// fragments: run u = k rows (2u, 2u+1) in b[u][0] and (2u+8, 2u+9) in b[u][1].
template <int K, int CB>
__device__ __forceinline__ void decode_tile_sm80(
    const LaneWords<K, false>& lw, const Window<K, false>& win,
    uint32_t (&b)[4][2]) {
  const int o0 = win.o[0] & 31;
  const uint32_t* w = lw.w;
#pragma unroll
  for (int u = 0; u < 4; ++u) {
    constexpr int kStep = 8 * K;
    const int ml = (kStep * u) >> 5, ol = (kStep * u) & 31;
    const int ou = o0 + ol;
    const bool carry = ou >= 32;
    decode_run<K, 4, CB>(carry ? w[ml + 1] : w[ml],
                         carry ? w[ml + 2] : w[ml + 1], ou & 31, b[u]);
  }
}

// One warp's share of one row tile: y[0 .. rows) x [n0, n0 + 32) of x @ T
// over k-tiles [tk0, tk1) (split-K: the caller offsets y to its partial),
// x / y pointing at the tile's first row. P = k-tiles whose words are in flight.
template <int K, int MT, int CB, int P>
__device__ __forceinline__ void tiled_accum(const uint32_t* __restrict__ tr,
                                            const half* __restrict__ x,
                                            int rows, int k, int n0, int tk0,
                                            int tk1, float (&d)[MT / 8][8]) {
  constexpr int NG = MT / 8;
  constexpr int kTW = 8 * K;
  const int lane = threadIdx.x & 31;
  const int q = (lane >> 2) & 3;
  const int nw = 8 * q + (lane >> 4) * 4 + lane % 4;
  const int t = nw >> 4, c = nw & 15;
  const int row = (lane >> 4) * 4 + lane % 4;  // A row of this lane in a group
  const int ng = (rows + 7) >> 3;
  const int ktiles = k / 16;
  Window<K, false> win(t, c);
  // Rows past the tile re-read the last row; their outputs are never stored.
  int xo[NG];
#pragma unroll
  for (int g = 0; g < NG; ++g) xo[g] = min(8 * g + row, rows - 1) * k;
#pragma unroll
  for (int g = 0; g < NG; ++g) {
#pragma unroll
    for (int i = 0; i < 8; ++i) d[g][i] = 0.f;
  }
  // Ring of P k-tiles in flight: a slot is refilled with tile tk + P as soon
  // as tile tk is decoded, so the loads overlap this tile's MMA work.
  LaneWords<K, false> lw[P];
#pragma unroll
  for (int p = 0; p < P; ++p) {
    if (tk0 + p < tk1) {
      lw[p].load(strip_block(tr, n0, tk0 + p, ktiles, kTW), win);
    } else {
      lw[p].zero();
    }
  }
  for (int it0 = tk0; it0 < tk1; it0 += P) {
#pragma unroll
    for (int p = 0; p < P; ++p) {
      const int tk = it0 + p;
      if (tk >= tk1) break;
      uint32_t b[4][2];
      decode_tile_sm80<K, CB>(lw[p], win, b);
      if (tk + P < tk1) lw[p].load(strip_block(tr, n0, tk + P, ktiles, kTW), win);
      const half* xk = x + tk * 16;
#pragma unroll
      for (int g = 0; g < NG; ++g) {
        if (g < ng) {  // warp-uniform
          const uint4 lo = *reinterpret_cast<const uint4*>(xk + xo[g]);
          const uint4 hi = *reinterpret_cast<const uint4*>(xk + xo[g] + 8);
          const uint32_t xw[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
#pragma unroll
          for (int u = 0; u < 4; ++u) {
            uint32_t a[2] = {xw[u], xw[u + 4]};
            mma884(d[g], a, b[u]);
          }
        }
      }
    }
  }
}

template <int K, int MT, int CB, int P>
__device__ __forceinline__ void tiled_body(const uint32_t* __restrict__ tr,
                                           const half* __restrict__ x,
                                           float* __restrict__ y, int rows,
                                           int k, int n, int n0, int tk0,
                                           int tk1) {
  constexpr int NG = MT / 8;
  const int lane = threadIdx.x & 31;
  const int q = (lane >> 2) & 3;
  const int ng = (rows + 7) >> 3;
  float d[NG][8];
  tiled_accum<K, MT, CB, P>(tr, x, rows, k, n0, tk0, tk1, d);
  // D fragment: lane L, reg i -> row (L&1) + ((i>>1)&1)*2 + (L>=16 ? 4 : 0),
  // col ((i>>2)&1)*4 + (L&2) + (i&1) of quad pair q's 8 columns.
#pragma unroll
  for (int g = 0; g < NG; ++g) {
    if (g >= ng) break;
#pragma unroll
    for (int i = 0; i < 8; i += 2) {
      const int r = 8 * g + (lane & 1) + ((i >> 1) & 1) * 2 + (lane >= 16 ? 4 : 0);
      if (r < rows) {
        const int col = n0 + 8 * q + ((i >> 2) & 1) * 4 + (lane & 2);
        *reinterpret_cast<float2*>(y + static_cast<int64_t>(r) * n + col) =
            make_float2(d[g][i], d[g][i + 1]);
      }
    }
  }
}

#define EXL3_K_SWITCH_SM80(KB, CALL) \
  switch (KB) {                      \
    case 2: CALL(2); break;          \
    case 3: CALL(3); break;          \
    case 4: CALL(4); break;          \
    case 5: CALL(5); break;          \
    default: break;                  \
  }

// grid (ceil(n / kTileCols), NT or 2NT, ksplit). tiles[NT][4] = (expert, row0,
// rows, -): rows [row0, row0 + rows) of x / y (compacted, sorted by expert)
// multiply expert `expert`. rows <= MT; rows = 0 or expert < 0 is an empty
// tile. Split z covers k-tiles [z, z + 1) * k/16 / ksplit and writes partial z
// at y + z * split_stride (mid / post add the partials in fixed order).
template <int MT, int CB, int P, int MINB>
__global__ void __launch_bounds__(kWarps * 32, MINB) exl3_tiled_mixed_kernel(
    const uint32_t* __restrict__ trellis, const int64_t* __restrict__ offs,
    const int* __restrict__ kbits, const int* __restrict__ tiles,
    const half* __restrict__ x, float* __restrict__ y, int k, int n, int NT,
    const uint32_t* __restrict__ trellis2, const half* __restrict__ x2,
    float* __restrict__ y2, int64_t split_stride) {
  int tile = blockIdx.y;
  if (tile >= NT) {  // second projection of a fused gate + up launch
    tile -= NT;
    trellis = trellis2;
    x = x2;
    y = y2;
  }
  const int e = tiles[4 * tile], row0 = tiles[4 * tile + 1];
  const int rows = min(tiles[4 * tile + 2], MT);
  if (e < 0 || rows <= 0) return;
  const int n0 = (blockIdx.x * kWarps + (threadIdx.x >> 5)) * 32;
  if (n0 >= n) return;  // whole warp; there is no block barrier
  const uint32_t* tr = trellis + offs[e];
  const half* xt = x + static_cast<int64_t>(row0) * k;
  float* yt = y + blockIdx.z * split_stride + static_cast<int64_t>(row0) * n;
  const int span = k / 16 / gridDim.z;
  const int tk0 = blockIdx.z * span;
#define EXL3_TILED(KK) \
  tiled_body<KK, MT, CB, P>(tr, xt, yt, rows, k, n, n0, tk0, tk0 + span)
  EXL3_K_SWITCH_SM80(kbits[e], EXL3_TILED)
#undef EXL3_TILED
}

// Layout 1 (k-split warps): grid (n / 32, NT or 2NT, ksplit). The block's 4
// warps share one 32-column strip and split its k range in 4 (the 8-slot
// kernels' layout: 4x the blocks of layout 0 and a quarter of the serial
// k-walk per warp); the partial sums meet in shared memory per 8-row group.
template <int K, int MT, int CB, int P>
__device__ __forceinline__ void tiled_kwarp_body(
    const uint32_t* __restrict__ tr, const half* __restrict__ x,
    float* __restrict__ y, int rows, int k, int n, int n0, int tk0, int tk1) {
  constexpr int NG = MT / 8;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int q = (lane >> 2) & 3;
  const int ng = (rows + 7) >> 3;
  const int per = (tk1 - tk0) / kWarps;
  float d[NG][8];
  tiled_accum<K, MT, CB, P>(tr, x, rows, k, n0, tk0 + warp * per,
                            tk0 + (warp + 1) * per, d);
  __shared__ float red[kWarps][8][32];
#pragma unroll
  for (int g = 0; g < NG; ++g) {
    if (g >= ng) break;  // block-uniform
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      red[warp][(lane & 1) + ((i >> 1) & 1) * 2 + (lane >= 16 ? 4 : 0)]
         [8 * q + ((i >> 2) & 1) * 4 + (lane & 2) + (i & 1)] = d[g][i];
    }
    __syncthreads();
    for (int e = threadIdx.x; e < 256; e += kWarps * 32) {
      const int m = e >> 5, nn = e & 31, r = 8 * g + m;
      if (r < rows) {
        float acc = 0.f;
#pragma unroll
        for (int w = 0; w < kWarps; ++w) acc += red[w][m][nn];
        y[static_cast<int64_t>(r) * n + n0 + nn] = acc;
      }
    }
    __syncthreads();
  }
}

template <int MT, int CB, int P, int MINB>
__global__ void __launch_bounds__(kWarps * 32, MINB) exl3_tiled_kwarp_kernel(
    const uint32_t* __restrict__ trellis, const int64_t* __restrict__ offs,
    const int* __restrict__ kbits, const int* __restrict__ tiles,
    const half* __restrict__ x, float* __restrict__ y, int k, int n, int NT,
    const uint32_t* __restrict__ trellis2, const half* __restrict__ x2,
    float* __restrict__ y2, int64_t split_stride) {
  int tile = blockIdx.y;
  if (tile >= NT) {
    tile -= NT;
    trellis = trellis2;
    x = x2;
    y = y2;
  }
  const int e = tiles[4 * tile], row0 = tiles[4 * tile + 1];
  const int rows = min(tiles[4 * tile + 2], MT);
  if (e < 0 || rows <= 0) return;  // block-uniform
  const int n0 = blockIdx.x * 32;
  const uint32_t* tr = trellis + offs[e];
  const half* xt = x + static_cast<int64_t>(row0) * k;
  float* yt = y + blockIdx.z * split_stride + static_cast<int64_t>(row0) * n;
  const int span = k / 16 / gridDim.z;
  const int tk0 = blockIdx.z * span;
#define EXL3_TILED_KW(KK) \
  tiled_kwarp_body<KK, MT, CB, P>(tr, xt, yt, rows, k, n, n0, tk0, tk0 + span)
  EXL3_K_SWITCH_SM80(kbits[e], EXL3_TILED_KW)
#undef EXL3_TILED_KW
}

// Single block. loc[S] = rank-local expert of slot s (-1 = elsewhere). Counts
// the rows of each expert, lays them out by expert (row r -> slot row_slot[r],
// expert row_exp[r]; slot s -> row slot_row[s] or -1) and cuts each expert into
// row tiles: <= t16 rows -> one 16-row tile, <= t32 -> one 32-row tile, else
// 64-row tiles (tier 2). With dense_min > 0 an expert of >= dense_min rows gets
// no tiles and is listed in info[8 + 4d ..] = (expert, row0, rows, 0) for the
// reconstruct + GEMM path. info[0] = R, info[1..3] = tiles per tier, info[4] =
// dense experts. Rows of one expert land in atomic (arbitrary) order, which no
// result depends on (see above).
__global__ void __launch_bounds__(kPlanThreads) exl3_moe_plan_kernel(
    const int* __restrict__ loc, int S, int E, int t16, int t32, int dense_min,
    int ntmax, int* __restrict__ row_slot, int* __restrict__ slot_row,
    int* __restrict__ row_exp, int* __restrict__ tiles, int* __restrict__ info) {
  extern __shared__ int plan_sh[];
  int* cnt = plan_sh;
  int* off = plan_sh + E;
  int* fill = plan_sh + 2 * E;
  const int tid = threadIdx.x;
  for (int i = tid; i < E; i += blockDim.x) {
    cnt[i] = 0;
    fill[i] = 0;
  }
  for (int i = tid; i < S; i += blockDim.x) {
    row_slot[i] = -1;
    row_exp[i] = -1;
    slot_row[i] = -1;
  }
  for (int i = tid; i < kTiers * ntmax; i += blockDim.x) {
    *reinterpret_cast<int4*>(tiles + 4 * i) = make_int4(-1, 0, 0, 0);
  }
  __syncthreads();
  for (int s = tid; s < S; s += blockDim.x) {
    const int e = loc[s];
    if (e >= 0 && e < E) atomicAdd(&cnt[e], 1);
  }
  __syncthreads();
  if (tid == 0) {
    int r = 0, nd = 0, nt[kTiers] = {0, 0, 0};
    for (int e = 0; e < E; ++e) {
      const int c = cnt[e];
      off[e] = r;
      if (c > 0) {
        if (dense_min > 0 && c >= dense_min) {
          *reinterpret_cast<int4*>(info + kInfoHead + 4 * nd) = make_int4(e, r, c, 0);
          ++nd;
        } else {
          const int tier = c <= t16 ? 0 : (c <= t32 ? 1 : 2);
          const int mt = 16 << tier;
          for (int o = 0; o < c; o += mt) {
            *reinterpret_cast<int4*>(tiles + 4 * (tier * ntmax + nt[tier]++)) =
                make_int4(e, r + o, min(mt, c - o), 0);
          }
        }
      }
      r += c;
    }
    info[0] = r;
    info[1] = nt[0];
    info[2] = nt[1];
    info[3] = nt[2];
    info[4] = nd;
    info[5] = info[6] = info[7] = 0;
  }
  __syncthreads();
  for (int s = tid; s < S; s += blockDim.x) {
    const int e = loc[s];
    if (e < 0 || e >= E) continue;
    const int r = off[e] + atomicAdd(&fill[e], 1);
    row_slot[r] = s;
    row_exp[r] = e;
    slot_row[s] = r;
  }
}

// Row r: X'[r] = had128(x[row_slot[r] / topk] * suh[row_exp[r]]).
__global__ void exl3_pre_rows_kernel(const half* __restrict__ x,
                                     const int* __restrict__ row_slot,
                                     const int* __restrict__ row_exp,
                                     const half* __restrict__ suh,
                                     half* __restrict__ out, int k, int topk,
                                     int R, const half* __restrict__ suh2,
                                     half* __restrict__ out2) {
  int r = blockIdx.y;
  if (r >= R) {
    r -= R;
    suh = suh2;
    out = out2;
  }
  const int e = row_exp[r];
  if (e < 0) return;
  int o = glue_offset();
  if (o >= k) return;
  float v[4], g[4];
  load4h(x + static_cast<int64_t>(row_slot[r] / topk) * k + o, v);
  load4h(suh + static_cast<int64_t>(e) * k + o, g);
#pragma unroll
  for (int i = 0; i < 4; ++i) v[i] *= g[i];
  fwht128(v);
  store4h(out + static_cast<int64_t>(r) * k + o, v);
}

// Slot s: out[s] = fp16(had128(yd[slot_row[s]]) * svh_d[e]), 0 for a slot of
// another rank: the values exl3_post_kernel gives with weight 1, then .half()
// (up to the sign of a zero).
__global__ void exl3_post_rows_kernel(const float* __restrict__ yd,
                                      const int* __restrict__ slot_row,
                                      const int* __restrict__ row_exp,
                                      const half* __restrict__ svh_d,
                                      half* __restrict__ out, int n,
                                      int ksplit, int split_rows) {
  const int s = blockIdx.y;
  const int o = glue_offset();
  if (o >= n) return;
  const int r = slot_row[s];
  float v[4] = {0.f, 0.f, 0.f, 0.f};
  if (r >= 0) {  // warp-uniform
    for (int z = 0; z < ksplit; ++z) {  // split-K partials, fixed order
      const float4 a = *reinterpret_cast<const float4*>(
          yd + (static_cast<int64_t>(z) * split_rows + r) * n + o);
      v[0] += a.x;
      v[1] += a.y;
      v[2] += a.z;
      v[3] += a.w;
    }
    fwht128(v);
    float sv[4];
    load4h(svh_d + static_cast<int64_t>(row_exp[r]) * n + o, sv);
#pragma unroll
    for (int i = 0; i < 4; ++i) v[i] *= sv[i];
  }
  store4h(out + static_cast<int64_t>(s) * n + o, v);
}

// Reconstruct one expert's trellis-basis matrix T [k, n] (fp16, row-major)
// from its strip-major trellis: grid (ceil(n / kTileCols), ceil(k/16 /
// kReconTiles)), warp = one 32-column strip, kReconTiles k-tiles.
template <int K, int CB>
__device__ __forceinline__ void recon_body(const uint32_t* __restrict__ tr,
                                           half* __restrict__ w, int k, int n,
                                           int n0, int tk0) {
  constexpr int kTW = 8 * K;
  const int lane = threadIdx.x & 31;
  const int q = (lane >> 2) & 3;
  const int nw = 8 * q + (lane >> 4) * 4 + lane % 4;
  const int t = nw >> 4, c = nw & 15;
  const int ktiles = k / 16;
  Window<K, false> win(t, c);
  half* wc = w + n0 + nw;
#pragma unroll 2
  for (int p = 0; p < kReconTiles; ++p) {
    const int tk = tk0 + p;
    if (tk >= ktiles) break;
    LaneWords<K, false> lw;
    lw.load(strip_block(tr, n0, tk, ktiles, kTW), win);
    uint32_t b[4][2];
    decode_tile_sm80<K, CB>(lw, win, b);
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      const half2 lo = *reinterpret_cast<const half2*>(&b[u][0]);
      const half2 hi = *reinterpret_cast<const half2*>(&b[u][1]);
      half* base = wc + static_cast<int64_t>(tk * 16 + 2 * u) * n;
      base[0] = __low2half(lo);
      base[n] = __high2half(lo);
      base[8 * static_cast<int64_t>(n)] = __low2half(hi);
      base[9 * static_cast<int64_t>(n)] = __high2half(hi);
    }
  }
}

template <int CB>
__global__ void __launch_bounds__(kWarps * 32) exl3_reconstruct_kernel(
    const uint32_t* __restrict__ tr, half* __restrict__ w, int k, int n, int K) {
  const int n0 = (blockIdx.x * kWarps + (threadIdx.x >> 5)) * 32;
  if (n0 >= n) return;
  const int tk0 = blockIdx.y * kReconTiles;
#define EXL3_RECON(KK) recon_body<KK, CB>(tr, w, k, n, n0, tk0)
  EXL3_K_SWITCH_SM80(K, EXL3_RECON)
#undef EXL3_RECON
}
#undef EXL3_K_SWITCH_SM80

template <int MT, int CB>
void launch_tiled(int layout, dim3 grid, cudaStream_t stream, const uint32_t* tr,
                  const int64_t* op, const int* kp, const int* tiles,
                  const half* x, float* y, int k, int n, int NT,
                  const uint32_t* tr2, const half* x2, float* y2,
                  int64_t split_stride) {
  // P = k-tiles whose words are in flight per warp, MINB = blocks per SM the
  // register budget is cut for (the 64-row tile holds 64 accumulators). The
  // defaults are the largest no-spill (mul1) budgets at 16 / 12 / 12 warps per
  // SM (127 / 164 / 168 registers); -D overrides are for the tuning builds.
  constexpr int P = MT >= 64 ? EXL3_TILED_P64 : (MT >= 32 ? EXL3_TILED_P32 : EXL3_TILED_P16);
  constexpr int MINB = MT >= 64 ? EXL3_TILED_MINB64 : (MT >= 32 ? EXL3_TILED_MINB32 : EXL3_TILED_MINB16);
  if (layout == 1) {
    constexpr int PK = MT >= 64 ? EXL3_TILED_KW_P64 : (MT >= 32 ? EXL3_TILED_KW_P32 : EXL3_TILED_KW_P16);
    exl3_tiled_kwarp_kernel<MT, CB, PK, MINB><<<grid, kWarps * 32, 0, stream>>>(
        tr, op, kp, tiles, x, y, k, n, NT, tr2, x2, y2, split_stride);
  } else {
    exl3_tiled_mixed_kernel<MT, CB, P, MINB><<<grid, kWarps * 32, 0, stream>>>(
        tr, op, kp, tiles, x, y, k, n, NT, tr2, x2, y2, split_stride);
  }
}

}  // namespace

// xg = had128(x[t] * g_suh[e]) per slot; with out2/suh2 also the up input.
void exl3_moe_pre_out(torch::Tensor out, std::optional<torch::Tensor> out2,
                      torch::Tensor x, torch::Tensor topk_ids,
                      torch::Tensor suh, std::optional<torch::Tensor> suh2) {
  check_cuda(x, "x", at::kHalf, 2, x);
  check_cuda(topk_ids, "topk_ids", at::kInt, 2, x);
  check_cuda(out, "out", at::kHalf, 2, x);
  check_cuda(suh, "suh", at::kHalf, 2, x);
  TORCH_CHECK(out2.has_value() == suh2.has_value(),
              "out2 and suh2 go together");
  const int k = static_cast<int>(x.size(1));
  const int topk = static_cast<int>(topk_ids.size(1));
  const int S = static_cast<int>(topk_ids.numel());
  TORCH_CHECK(topk_ids.size(0) == x.size(0), "topk_ids rows != tokens");
  TORCH_CHECK(k % 128 == 0, "EXL3 MoE: k must be a multiple of 128");
  TORCH_CHECK(out.size(0) >= S && out.size(1) == k && suh.size(1) == k,
              "pre: shape mismatch");
  const int fused = out2.has_value() ? 2 : 1;
  if (out2) {
    check_cuda(*out2, "out2", at::kHalf, 2, x);
    check_cuda(*suh2, "suh2", at::kHalf, 2, x);
    TORCH_CHECK(out2->size(0) >= S && out2->size(1) == k && suh2->size(1) == k,
                "pre: shape mismatch");
  }
  if (S == 0) return;
  TORCH_CHECK(fused * S <= kMaxGridY, "EXL3 MoE: too many slots per launch");
  const at::cuda::OptionalCUDAGuard guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  exl3_pre_kernel<<<dim3(glue_blocks(k), fused * S), kGlueWarps * 32, 0,
                    stream>>>(
      reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
      topk_ids.data_ptr<int>(),
      reinterpret_cast<const half*>(suh.data_ptr<at::Half>()),
      reinterpret_cast<half*>(out.data_ptr<at::Half>()), k, topk, S,
      out2 ? reinterpret_cast<const half*>(suh2->data_ptr<at::Half>())
           : nullptr,
      out2 ? reinterpret_cast<half*>(out2->data_ptr<at::Half>()) : nullptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// y[s] = x[s] @ T[expert(s)] in the trellis basis. Plain: expert_ids [S] holds
// each slot's expert. Grouped: expert_ids [G] holds each group's expert and
// group_rows [G, 8] its slots (-1 = empty). With y2/x2/trellis2, the second
// projection (up) runs in the same launch. codebook: 0 = mcg, 1 = mul1.
void exl3_moe_gemv_out(torch::Tensor y, std::optional<torch::Tensor> y2,
                       torch::Tensor x, std::optional<torch::Tensor> x2,
                       torch::Tensor trellis,
                       std::optional<torch::Tensor> trellis2,
                       torch::Tensor expert_ids,
                       std::optional<torch::Tensor> group_rows, bool colmajor,
                       int64_t codebook) {
  TORCH_CHECK(codebook == kCodebookMcg || codebook == kCodebookMul1,
              "EXL3 MoE: codebook must be 0 (mcg) or 1 (mul1)");
  check_cuda(x, "x", at::kHalf, 2, x);
  check_cuda(y, "y", at::kFloat, 2, x);
  check_cuda(expert_ids, "expert_ids", at::kInt, 1, x);
  auto [k, n, K] = trellis_dims(trellis, x);
  TORCH_CHECK(!(colmajor && K == 5), "EXL3 MoE: K = 5 needs the sm80 tile order");
  TORCH_CHECK(x.size(1) == k && y.size(1) == n && y.size(0) >= x.size(0),
              "gemv: shape mismatch");
  const bool fused = y2.has_value();
  TORCH_CHECK(fused == x2.has_value() && fused == trellis2.has_value(),
              "y2, x2 and trellis2 go together");
  if (fused) {
    check_cuda(*y2, "y2", at::kFloat, 2, x);
    check_cuda(*x2, "x2", at::kHalf, 2, x);
    auto [k2, n2, K2] = trellis_dims(*trellis2, x);
    TORCH_CHECK(k2 == k && n2 == n && K2 == K &&
                    trellis2->size(0) == trellis.size(0) &&
                    x2->sizes() == x.sizes() && y2->sizes() == y.sizes(),
                "gemv: the fused projections must have the same shapes");
  }
  const bool grouped = group_rows.has_value();
  int count = static_cast<int>(expert_ids.size(0));
  if (grouped) {
    check_cuda(*group_rows, "group_rows", at::kInt, 2, x);
    TORCH_CHECK(group_rows->size(0) == count && group_rows->size(1) == 8,
                "group_rows must be [G, 8]");
  } else {
    TORCH_CHECK(count <= x.size(0), "expert_ids longer than x");
  }
  if (count == 0) return;
  TORCH_CHECK((fused ? 2 : 1) * count <= kMaxGridY,
              "EXL3 MoE: too many slots per launch");
  const int64_t ew = trellis[0].numel() / 2;  // u32 words per expert
  const at::cuda::OptionalCUDAGuard guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 grid(n / 32, (fused ? 2 : 1) * count);
  auto tr = reinterpret_cast<const uint32_t*>(trellis.data_ptr<int16_t>());
  auto tr2 =
      fused ? reinterpret_cast<const uint32_t*>(trellis2->data_ptr<int16_t>())
            : nullptr;
  auto xp = reinterpret_cast<const half*>(x.data_ptr<at::Half>());
  auto x2p =
      fused ? reinterpret_cast<const half*>(x2->data_ptr<at::Half>()) : nullptr;
  float* y2p = fused ? y2->data_ptr<float>() : nullptr;
  const int* rows = grouped ? group_rows->data_ptr<int>() : nullptr;
  const int cb = static_cast<int>(codebook);
  if (colmajor) {
    launch_gemv_cb<true>(cb, grouped, K, grid, stream, tr,
                         expert_ids.data_ptr<int>(), rows, xp,
                         y.data_ptr<float>(), k, n, ew, count, tr2, x2p, y2p);
  } else {
    launch_gemv_cb<false>(cb, grouped, K, grid, stream, tr,
                          expert_ids.data_ptr<int>(), rows, xp,
                          y.data_ptr<float>(), k, n, ew, count, tr2, x2p, y2p);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Mixed-K variant of exl3_moe_gemv_out: trellis (and trellis2) are flat int16
// buffers holding every expert's strip-major trellis; expert e starts at u32
// word offsets[e] and has kbits[e] (2..5) bits per weight. n = out_features.
void exl3_moe_gemv_mixed_out(torch::Tensor y, std::optional<torch::Tensor> y2,
                             torch::Tensor x, std::optional<torch::Tensor> x2,
                             torch::Tensor trellis,
                             std::optional<torch::Tensor> trellis2,
                             torch::Tensor offsets, torch::Tensor kbits,
                             torch::Tensor expert_ids,
                             std::optional<torch::Tensor> group_rows,
                             bool colmajor, int64_t codebook, int64_t n64,
                             int64_t ksplit) {
  TORCH_CHECK(codebook == kCodebookMcg || codebook == kCodebookMul1,
              "EXL3 MoE: codebook must be 0 (mcg) or 1 (mul1)");
  check_cuda(x, "x", at::kHalf, 2, x);
  check_cuda(y, "y", at::kFloat, 2, x);
  check_cuda(trellis, "trellis", at::kShort, 1, x);
  check_cuda(offsets, "offsets", at::kLong, 1, x);
  check_cuda(kbits, "kbits", at::kInt, 1, x);
  check_cuda(expert_ids, "expert_ids", at::kInt, 1, x);
  TORCH_CHECK(offsets.size(0) == kbits.size(0), "offsets and kbits: one per expert");
  const int k = static_cast<int>(x.size(1)), n = static_cast<int>(n64);
  TORCH_CHECK(ksplit >= 1 && k % 16 == 0 && (k / 16) % (kWarps * ksplit) == 0 && n % 32 == 0,
              "EXL3 MoE: k must be a multiple of 64 * ksplit and n of 32");
  TORCH_CHECK(y.size(1) == n && y.size(0) % ksplit == 0 && y.size(0) / ksplit >= x.size(0),
              "gemv: y must hold ksplit x rows partials");
  const int64_t split_stride = (y.size(0) / ksplit) * static_cast<int64_t>(n);
  const bool fused = y2.has_value();
  TORCH_CHECK(fused == x2.has_value() && fused == trellis2.has_value(),
              "y2, x2 and trellis2 go together");
  if (fused) {
    check_cuda(*y2, "y2", at::kFloat, 2, x);
    check_cuda(*x2, "x2", at::kHalf, 2, x);
    check_cuda(*trellis2, "trellis2", at::kShort, 1, x);
    TORCH_CHECK(trellis2->numel() == trellis.numel() && x2->sizes() == x.sizes() &&
                    y2->sizes() == y.sizes(),
                "gemv: the fused projections must have the same shapes");
  }
  const bool grouped = group_rows.has_value();
  const int count = static_cast<int>(expert_ids.size(0));
  if (grouped) {
    check_cuda(*group_rows, "group_rows", at::kInt, 2, x);
    TORCH_CHECK(group_rows->size(0) == count && group_rows->size(1) == 8,
                "group_rows must be [G, 8]");
    TORCH_CHECK(ksplit == 1, "EXL3 MoE: the grouped (prefill) path has no split-K");
  } else {
    TORCH_CHECK(count <= x.size(0), "expert_ids longer than x");
  }
  if (count == 0) return;
  TORCH_CHECK((fused ? 2 : 1) * count <= kMaxGridY,
              "EXL3 MoE: too many slots per launch");
  const at::cuda::OptionalCUDAGuard guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 grid(n / 32, (fused ? 2 : 1) * count, static_cast<unsigned>(ksplit));
  const int kt = k / 16 / kWarps / static_cast<int>(ksplit);
  auto tr = reinterpret_cast<const uint32_t*>(trellis.data_ptr<int16_t>());
  auto tr2 = fused
                 ? reinterpret_cast<const uint32_t*>(trellis2->data_ptr<int16_t>())
                 : nullptr;
  auto xp = reinterpret_cast<const half*>(x.data_ptr<at::Half>());
  auto x2p = fused ? reinterpret_cast<const half*>(x2->data_ptr<at::Half>()) : nullptr;
  float* y2p = fused ? y2->data_ptr<float>() : nullptr;
  const int64_t* op = offsets.data_ptr<int64_t>();
  const int* kp = kbits.data_ptr<int>();
  const int* ids = expert_ids.data_ptr<int>();
#define EXL3_MIXED(COL, CBV)                                                  \
  if (grouped) {                                                              \
    exl3_gemv_mixed_grouped_kernel<COL, CBV><<<grid, kWarps * 32, 0, stream>>>( \
        tr, op, kp, ids, group_rows->data_ptr<int>(), xp, y.data_ptr<float>(), \
        k, n, kt, count, tr2, x2p, y2p);                                      \
  } else {                                                                    \
    exl3_gemv_mixed_kernel<COL, CBV><<<grid, kWarps * 32, 0, stream>>>(       \
        tr, op, kp, ids, xp, y.data_ptr<float>(), k, n, kt, count, tr2, x2p,  \
        y2p, split_stride);                                                   \
  }
  if (colmajor) {
    if (codebook == kCodebookMul1) { EXL3_MIXED(true, kCodebookMul1) }
    else { EXL3_MIXED(true, kCodebookMcg) }
  } else {
    if (codebook == kCodebookMul1) { EXL3_MIXED(false, kCodebookMul1) }
    else { EXL3_MIXED(false, kCodebookMcg) }
  }
#undef EXL3_MIXED
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// out[s] = had128(silu(had128(yg) * g_svh) * had128(yu) * u_svh * d_suh); with
// limit > 0 the gate is clamped to <= limit and the up value to [-limit, limit].
void exl3_moe_mid_out(torch::Tensor out, torch::Tensor yg, torch::Tensor yu,
                      torch::Tensor topk_ids, torch::Tensor g_svh,
                      torch::Tensor u_svh, torch::Tensor d_suh, double limit,
                      int64_t ksplit) {
  check_cuda(yg, "yg", at::kFloat, 2, yg);
  check_cuda(yu, "yu", at::kFloat, 2, yg);
  check_cuda(out, "out", at::kHalf, 2, yg);
  check_cuda(topk_ids, "topk_ids", at::kInt, 2, yg);
  check_cuda(g_svh, "g_svh", at::kHalf, 2, yg);
  check_cuda(u_svh, "u_svh", at::kHalf, 2, yg);
  check_cuda(d_suh, "d_suh", at::kHalf, 2, yg);
  const int n = static_cast<int>(yg.size(1));
  const int S = static_cast<int>(topk_ids.numel());
  TORCH_CHECK(n % 128 == 0, "EXL3 MoE: n must be a multiple of 128");
  TORCH_CHECK(ksplit >= 1 && yg.size(0) % ksplit == 0, "mid: yg rows must be ksplit x split_rows");
  const int split_rows = static_cast<int>(yg.size(0) / ksplit);
  TORCH_CHECK(split_rows >= S && yu.sizes() == yg.sizes() && out.size(0) >= S &&
                  out.size(1) == n && g_svh.size(1) == n &&
                  u_svh.size(1) == n && d_suh.size(1) == n,
              "mid: shape mismatch");
  if (S == 0) return;
  TORCH_CHECK(S <= kMaxGridY, "EXL3 MoE: too many slots per launch");
  const at::cuda::OptionalCUDAGuard guard(yg.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  exl3_mid_kernel<<<dim3(glue_blocks(n), S), kGlueWarps * 32, 0, stream>>>(
      yg.data_ptr<float>(), yu.data_ptr<float>(), topk_ids.data_ptr<int>(),
      reinterpret_cast<const half*>(g_svh.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(u_svh.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(d_suh.data_ptr<at::Half>()),
      reinterpret_cast<half*>(out.data_ptr<at::Half>()), n,
      static_cast<float>(limit), static_cast<int>(ksplit), split_rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// out[t] = sum_j w[t, j] * had128(yd[t * topk + j]) * d_svh[e]   (fp32 [T, k])
void exl3_moe_post_out(torch::Tensor out, torch::Tensor yd,
                       torch::Tensor topk_ids, torch::Tensor topk_weights,
                       torch::Tensor d_svh, int64_t ksplit) {
  check_cuda(yd, "yd", at::kFloat, 2, yd);
  check_cuda(out, "out", at::kFloat, 2, yd);
  check_cuda(topk_ids, "topk_ids", at::kInt, 2, yd);
  check_cuda(topk_weights, "topk_weights", at::kFloat, 2, yd);
  check_cuda(d_svh, "d_svh", at::kHalf, 2, yd);
  const int k = static_cast<int>(yd.size(1));
  const int T = static_cast<int>(topk_ids.size(0));
  const int topk = static_cast<int>(topk_ids.size(1));
  TORCH_CHECK(k % 128 == 0, "EXL3 MoE: k must be a multiple of 128");
  TORCH_CHECK(ksplit >= 1 && yd.size(0) % ksplit == 0, "post: yd rows must be ksplit x split_rows");
  const int split_rows = static_cast<int>(yd.size(0) / ksplit);
  TORCH_CHECK(topk_weights.sizes() == topk_ids.sizes() &&
                  split_rows >= T * topk && out.size(0) >= T &&
                  out.size(1) == k && d_svh.size(1) == k,
              "post: shape mismatch");
  if (T == 0) return;
  TORCH_CHECK(T <= kMaxGridY, "EXL3 MoE: too many tokens per launch");
  const at::cuda::OptionalCUDAGuard guard(yd.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  exl3_post_kernel<<<dim3(glue_blocks(k), T), kGlueWarps * 32, 0, stream>>>(
      yd.data_ptr<float>(), topk_ids.data_ptr<int>(),
      topk_weights.data_ptr<float>(),
      reinterpret_cast<const half*>(d_svh.data_ptr<at::Half>()),
      out.data_ptr<float>(), k, topk, static_cast<int>(ksplit), split_rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---- prefill (row-tiled) ops ----

// Routing plan of one prefill pass. loc [S] int32 = rank-local expert of each
// slot (-1 = another rank's). Writes row_slot / row_exp [S] (rows sorted by
// expert, -1 past R), slot_row [S] (-1 = not here), tiles [3, ntmax, 4] (per
// tier 16 / 32 / 64 rows: expert, row0, rows, 0; empty tiles expert -1) and
// info [8 + 4E] (R, tiles per tier, dense experts, then the dense list; see
// exl3_moe_plan_kernel). One block, no host sync; the caller reads info.
void exl3_moe_plan_out(torch::Tensor row_slot, torch::Tensor slot_row,
                       torch::Tensor row_exp, torch::Tensor tiles,
                       torch::Tensor info, torch::Tensor loc,
                       int64_t num_experts, int64_t t16, int64_t t32,
                       int64_t dense_min) {
  check_cuda(loc, "loc", at::kInt, 1, loc);
  check_cuda(row_slot, "row_slot", at::kInt, 1, loc);
  check_cuda(slot_row, "slot_row", at::kInt, 1, loc);
  check_cuda(row_exp, "row_exp", at::kInt, 1, loc);
  check_cuda(tiles, "tiles", at::kInt, 3, loc);
  check_cuda(info, "info", at::kInt, 1, loc);
  const int S = static_cast<int>(loc.size(0));
  const int E = static_cast<int>(num_experts);
  TORCH_CHECK(E >= 1 && E <= 4096, "plan: 1..4096 experts");
  TORCH_CHECK(t16 >= 0 && t16 <= 16 && t32 >= 0 && t32 <= 32 && dense_min >= 0,
              "plan: tier limits must be 0..16 and 0..32");
  TORCH_CHECK(row_slot.size(0) == S && slot_row.size(0) == S && row_exp.size(0) == S,
              "plan: row_slot / slot_row / row_exp must have one entry per slot");
  const int ntmax = static_cast<int>(tiles.size(1));
  TORCH_CHECK(tiles.size(0) == kTiers && tiles.size(2) == 4 && ntmax >= S / 64 + E + 1,
              "plan: tiles must be [3, >= S / 64 + E + 1, 4]");
  TORCH_CHECK(info.size(0) >= kInfoHead + 4 * E, "plan: info must hold 8 + 4E ints");
  const at::cuda::OptionalCUDAGuard guard(loc.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  exl3_moe_plan_kernel<<<1, kPlanThreads, 3 * E * sizeof(int), stream>>>(
      loc.data_ptr<int>(), S, E, static_cast<int>(t16), static_cast<int>(t32),
      static_cast<int>(dense_min), ntmax, row_slot.data_ptr<int>(),
      slot_row.data_ptr<int>(), row_exp.data_ptr<int>(), tiles.data_ptr<int>(),
      info.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// out[r] = had128(x[row_slot[r] / topk] * suh[row_exp[r]]) for the R =
// row_slot.size(0) rows (row_exp < 0: skipped); with out2 / suh2 the up input.
void exl3_moe_pre_rows_out(torch::Tensor out, std::optional<torch::Tensor> out2,
                           torch::Tensor x, torch::Tensor row_slot,
                           torch::Tensor row_exp, torch::Tensor suh,
                           std::optional<torch::Tensor> suh2, int64_t topk) {
  check_cuda(x, "x", at::kHalf, 2, x);
  check_cuda(row_slot, "row_slot", at::kInt, 1, x);
  check_cuda(row_exp, "row_exp", at::kInt, 1, x);
  check_cuda(out, "out", at::kHalf, 2, x);
  check_cuda(suh, "suh", at::kHalf, 2, x);
  TORCH_CHECK(out2.has_value() == suh2.has_value(), "out2 and suh2 go together");
  const int k = static_cast<int>(x.size(1));
  const int R = static_cast<int>(row_slot.size(0));
  TORCH_CHECK(row_exp.size(0) == R && topk >= 1, "pre_rows: row_slot / row_exp mismatch");
  TORCH_CHECK(k % 128 == 0, "EXL3 MoE: k must be a multiple of 128");
  TORCH_CHECK(out.size(0) >= R && out.size(1) == k && suh.size(1) == k,
              "pre_rows: shape mismatch");
  const int fused = out2.has_value() ? 2 : 1;
  if (out2) {
    check_cuda(*out2, "out2", at::kHalf, 2, x);
    check_cuda(*suh2, "suh2", at::kHalf, 2, x);
    TORCH_CHECK(out2->size(0) >= R && out2->size(1) == k && suh2->size(1) == k,
                "pre_rows: shape mismatch");
  }
  if (R == 0) return;
  TORCH_CHECK(fused * R <= kMaxGridY, "EXL3 MoE: too many rows per launch");
  const at::cuda::OptionalCUDAGuard guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  exl3_pre_rows_kernel<<<dim3(glue_blocks(k), fused * R), kGlueWarps * 32, 0,
                         stream>>>(
      reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
      row_slot.data_ptr<int>(), row_exp.data_ptr<int>(),
      reinterpret_cast<const half*>(suh.data_ptr<at::Half>()),
      reinterpret_cast<half*>(out.data_ptr<at::Half>()), k,
      static_cast<int>(topk), R,
      out2 ? reinterpret_cast<const half*>(suh2->data_ptr<at::Half>()) : nullptr,
      out2 ? reinterpret_cast<half*>(out2->data_ptr<at::Half>()) : nullptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// y[rows of tile] = x[rows of tile] @ T[expert of tile] for every tile of one
// tier (tiles [NT, 4], row_tile = 16 / 32 / 64 rows), mixed K (offsets /
// kbits as exl3_moe_gemv_mixed_out), sm80 order. x [R, k] fp16, y [R, n] fp32
// (compacted rows); with y2 / x2 / trellis2 the up projection in the same
// launch. ksplit > 1: y holds ksplit partials of y.size(0) / ksplit rows (as
// exl3_moe_gemv_mixed_out). layout 0: block = 4 warps x 32 columns, each warp
// over all of k; layout 1: block = one 32-column strip, k split over 4 warps.
void exl3_moe_gemm_tiled_out(torch::Tensor y, std::optional<torch::Tensor> y2,
                             torch::Tensor x, std::optional<torch::Tensor> x2,
                             torch::Tensor trellis,
                             std::optional<torch::Tensor> trellis2,
                             torch::Tensor offsets, torch::Tensor kbits,
                             torch::Tensor tiles, int64_t row_tile,
                             int64_t codebook, int64_t n64, int64_t ksplit,
                             int64_t layout) {
  TORCH_CHECK(codebook == kCodebookMcg || codebook == kCodebookMul1,
              "EXL3 MoE: codebook must be 0 (mcg) or 1 (mul1)");
  TORCH_CHECK(row_tile == 16 || row_tile == 32 || row_tile == 64,
              "EXL3 MoE: row_tile must be 16, 32 or 64");
  check_cuda(x, "x", at::kHalf, 2, x);
  check_cuda(y, "y", at::kFloat, 2, x);
  check_cuda(trellis, "trellis", at::kShort, 1, x);
  check_cuda(offsets, "offsets", at::kLong, 1, x);
  check_cuda(kbits, "kbits", at::kInt, 1, x);
  check_cuda(tiles, "tiles", at::kInt, 2, x);
  TORCH_CHECK(offsets.size(0) == kbits.size(0), "offsets and kbits: one per expert");
  TORCH_CHECK(tiles.size(1) == 4, "tiles must be [NT, 4]");
  const int k = static_cast<int>(x.size(1)), n = static_cast<int>(n64);
  TORCH_CHECK(k % 64 == 0 && n % 32 == 0, "EXL3 MoE: k must be a multiple of 64 and n of 32");
  TORCH_CHECK(layout == 0 || layout == 1, "gemm_tiled: layout must be 0 or 1");
  TORCH_CHECK(ksplit >= 1 && (k / 16) % (ksplit * (layout == 1 ? kWarps : 1)) == 0,
              "gemm_tiled: ksplit (x 4 warps in layout 1) must divide k / 16");
  TORCH_CHECK(y.size(1) == n && y.size(0) % ksplit == 0 && y.size(0) / ksplit >= x.size(0),
              "gemm_tiled: y must hold ksplit x rows partials");
  const int64_t split_stride = (y.size(0) / ksplit) * static_cast<int64_t>(n);
  const bool fused = y2.has_value();
  TORCH_CHECK(fused == x2.has_value() && fused == trellis2.has_value(),
              "y2, x2 and trellis2 go together");
  if (fused) {
    check_cuda(*y2, "y2", at::kFloat, 2, x);
    check_cuda(*x2, "x2", at::kHalf, 2, x);
    check_cuda(*trellis2, "trellis2", at::kShort, 1, x);
    TORCH_CHECK(trellis2->numel() == trellis.numel() && x2->sizes() == x.sizes() &&
                    y2->sizes() == y.sizes(),
                "gemm_tiled: the fused projections must have the same shapes");
  }
  const int NT = static_cast<int>(tiles.size(0));
  if (NT == 0) return;
  TORCH_CHECK((fused ? 2 : 1) * NT <= kMaxGridY, "EXL3 MoE: too many tiles per launch");
  const at::cuda::OptionalCUDAGuard guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 grid(layout == 1 ? n / 32 : (n + kTileCols - 1) / kTileCols, (fused ? 2 : 1) * NT,
            static_cast<unsigned>(ksplit));
  auto tr = reinterpret_cast<const uint32_t*>(trellis.data_ptr<int16_t>());
  auto tr2 = fused ? reinterpret_cast<const uint32_t*>(trellis2->data_ptr<int16_t>())
                   : nullptr;
  auto xp = reinterpret_cast<const half*>(x.data_ptr<at::Half>());
  auto x2p = fused ? reinterpret_cast<const half*>(x2->data_ptr<at::Half>()) : nullptr;
  float* y2p = fused ? y2->data_ptr<float>() : nullptr;
  const int64_t* op = offsets.data_ptr<int64_t>();
  const int* kp = kbits.data_ptr<int>();
  const int* tp = tiles.data_ptr<int>();
#define EXL3_TILED_LAUNCH(MTV, CBV)                                             \
  launch_tiled<MTV, CBV>(static_cast<int>(layout), grid, stream, tr, op, kp, tp, xp, y.data_ptr<float>(), \
                         k, n, NT, tr2, x2p, y2p, split_stride)
#define EXL3_TILED_CB(MTV)                                                     \
  if (codebook == kCodebookMul1) {                                            \
    EXL3_TILED_LAUNCH(MTV, kCodebookMul1);                                    \
  } else {                                                                    \
    EXL3_TILED_LAUNCH(MTV, kCodebookMcg);                                     \
  }
  if (row_tile == 16) {
    EXL3_TILED_CB(16)
  } else if (row_tile == 32) {
    EXL3_TILED_CB(32)
  } else {
    EXL3_TILED_CB(64)
  }
#undef EXL3_TILED_CB
#undef EXL3_TILED_LAUNCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// out[s] = fp16(had128(yd[slot_row[s]]) * d_svh[row_exp[slot_row[s]]]) for
// every slot (S = slot_row.size(0)); 0 where slot_row < 0. yd holds ksplit
// partials of yd.size(0) / ksplit rows each.
void exl3_moe_post_rows_out(torch::Tensor out, torch::Tensor yd,
                            torch::Tensor slot_row, torch::Tensor row_exp,
                            torch::Tensor d_svh, int64_t ksplit) {
  check_cuda(yd, "yd", at::kFloat, 2, yd);
  check_cuda(out, "out", at::kHalf, 2, yd);
  check_cuda(slot_row, "slot_row", at::kInt, 1, yd);
  check_cuda(row_exp, "row_exp", at::kInt, 1, yd);
  check_cuda(d_svh, "d_svh", at::kHalf, 2, yd);
  const int n = static_cast<int>(yd.size(1));
  const int S = static_cast<int>(slot_row.size(0));
  TORCH_CHECK(n % 128 == 0, "EXL3 MoE: n must be a multiple of 128");
  TORCH_CHECK(out.size(0) >= S && out.size(1) == n && d_svh.size(1) == n,
              "post_rows: shape mismatch");
  TORCH_CHECK(ksplit >= 1 && yd.size(0) % ksplit == 0, "post_rows: yd rows must be ksplit x split_rows");
  const int split_rows = static_cast<int>(yd.size(0) / ksplit);
  if (S == 0) return;
  TORCH_CHECK(S <= kMaxGridY, "EXL3 MoE: too many slots per launch");
  const at::cuda::OptionalCUDAGuard guard(yd.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  exl3_post_rows_kernel<<<dim3(glue_blocks(n), S), kGlueWarps * 32, 0, stream>>>(
      yd.data_ptr<float>(), slot_row.data_ptr<int>(), row_exp.data_ptr<int>(),
      reinterpret_cast<const half*>(d_svh.data_ptr<at::Half>()),
      reinterpret_cast<half*>(out.data_ptr<at::Half>()), n, static_cast<int>(ksplit),
      split_rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// w [k, n] fp16 = the trellis-basis matrix of the expert whose strip-major
// trellis starts at u32 word `offset` of trellis (K bits, sm80 order).
void exl3_moe_reconstruct_out(torch::Tensor w, torch::Tensor trellis,
                              int64_t offset, int64_t K, int64_t codebook) {
  check_cuda(w, "w", at::kHalf, 2, w);
  check_cuda(trellis, "trellis", at::kShort, 1, w);
  TORCH_CHECK(K >= 2 && K <= 5, "EXL3 MoE: K must be 2..5");
  TORCH_CHECK(codebook == kCodebookMcg || codebook == kCodebookMul1,
              "EXL3 MoE: codebook must be 0 (mcg) or 1 (mul1)");
  const int k = static_cast<int>(w.size(0)), n = static_cast<int>(w.size(1));
  TORCH_CHECK(k % 16 == 0 && n % 32 == 0, "reconstruct: k must be a multiple of 16 and n of 32");
  TORCH_CHECK(offset >= 0 && (offset + static_cast<int64_t>(k) * n * K / 32) * 2 <= trellis.numel(),
              "reconstruct: expert outside the trellis buffer");
  const at::cuda::OptionalCUDAGuard guard(w.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 grid((n + kTileCols - 1) / kTileCols, (k / 16 + kReconTiles - 1) / kReconTiles);
  auto tr = reinterpret_cast<const uint32_t*>(trellis.data_ptr<int16_t>()) + offset;
  auto wp = reinterpret_cast<half*>(w.data_ptr<at::Half>());
  if (codebook == kCodebookMul1) {
    exl3_reconstruct_kernel<kCodebookMul1><<<grid, kWarps * 32, 0, stream>>>(
        tr, wp, k, n, static_cast<int>(K));
  } else {
    exl3_reconstruct_kernel<kCodebookMcg><<<grid, kWarps * 32, 0, stream>>>(
        tr, wp, k, n, static_cast<int>(K));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// y [m, n] fp32 = x [m, k] fp16 @ w [k, n] fp16 (cuBLAS, fp32 accumulate);
// all row-major and contiguous (row slices of the compacted buffers).
void exl3_moe_dense_out(torch::Tensor y, torch::Tensor x, torch::Tensor w) {
  check_cuda(x, "x", at::kHalf, 2, x);
  check_cuda(w, "w", at::kHalf, 2, x);
  check_cuda(y, "y", at::kFloat, 2, x);
  const int m = static_cast<int>(x.size(0)), k = static_cast<int>(x.size(1));
  const int n = static_cast<int>(w.size(1));
  TORCH_CHECK(w.size(0) == k && y.size(0) == m && y.size(1) == n, "dense: shape mismatch");
  if (m == 0) return;
  const at::cuda::OptionalCUDAGuard guard(x.device());
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();  // on the current stream
  const float alpha = 1.f, beta = 0.f;
  // column-major view: y^T [n, m] = w^T [n, k] @ x^T [k, m]
  TORCH_CUDABLAS_CHECK(cublasGemmEx(
      handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, w.data_ptr(), CUDA_R_16F, n,
      x.data_ptr(), CUDA_R_16F, k, &beta, y.data_ptr(), CUDA_R_32F, n,
      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}
