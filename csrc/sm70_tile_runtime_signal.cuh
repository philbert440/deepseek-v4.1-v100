#pragma once

#include <cstdint>

namespace vllm::sm70_tile_runtime {

// Keep the default TileRT/all-reduce block limit unchanged.  The larger signal
// capacity is reserved for long-prefill collective-fusion experiments that use
// one cross-rank barrier per owned token.
constexpr int kMaxBlocks = 64;
constexpr int kMaxSignalBlocks = 512;
constexpr int kMaxRanks = 8;

using FlagType = uint32_t;

struct Signal {
  alignas(128) FlagType start[kMaxSignalBlocks][kMaxRanks];
  alignas(128) FlagType end[kMaxSignalBlocks][kMaxRanks];
  alignas(128) FlagType _flag[kMaxSignalBlocks];
};

struct __align__(16) RankData {
  const void* ptrs[kMaxRanks];
};

struct __align__(16) RankSignals {
  Signal* signals[kMaxRanks];
};

#if !defined(USE_ROCM)
static __device__ __forceinline__ void store_flag_sys_visible(
    FlagType* flag_addr, FlagType flag) {
  asm volatile("membar.sys; st.volatile.global.u32 [%1], %0;" ::"r"(flag),
               "l"(flag_addr)
               : "memory");
}

static __device__ __forceinline__ FlagType
load_flag_sys_visible(FlagType* flag_addr) {
  FlagType flag;
  asm volatile("ld.volatile.global.u32 %0, [%1]; membar.sys;"
               : "=r"(flag)
               : "l"(flag_addr)
               : "memory");
  return flag;
}
#endif

}  // namespace vllm::sm70_tile_runtime
