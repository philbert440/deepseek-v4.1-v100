// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// DeepSeek-V4.1 Engram RAM-resident row tables (VLLM_DS41_ENGRAM_RESIDENT=1; lane L-ENGRAM, common/engram_host.py).
//
// With the knob on, each TP rank of the Engram stage keeps the 256-byte E4M3 rows of ITS OWN sub-tables in one
// anonymous mapping per layer (MADV_HUGEPAGE, optionally mbind-interleaved over NUMA nodes), filled once at init with
// large sequential preads from the official (sha256-verified) shards, laid out exactly like the rank's scale table
// (sub-tables concatenated in rank order). The per-step gather is then a memcpy per unique row:
//   dst[i] = rows[tab[i]][sidx[i]] (256 B) || scales[tab[i]][sidx[i]] (8 B)
// i.e. the same 264-byte record the pread path (csrc/engram_rows.cpp) writes, from the same file bytes. No I/O
// threads, no page cache / ARC after the load. The big tables are NOT page-locked (only the small staging is).
//
// Built at runtime by torch.utils.cpp_extension.load only when the knob is on (separate source on purpose: the
// default path keeps loading the unchanged engram_rows extension).

#include <torch/extension.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

#ifndef MPOL_INTERLEAVE
  #define MPOL_INTERLEAVE 3
#endif

namespace {

constexpr int64_t kAlign = 4096;

int64_t now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

// One exact-size anonymous mapping, THP-advised, optionally interleaved over `nodes` (empty: the task's policy).
// Freed (munmap) when the last tensor referencing it goes away.
torch::Tensor alloc_resident(int64_t nbytes, const std::vector<int64_t>& nodes, bool hugepage) {
  TORCH_CHECK(nbytes > 0, "Engram resident: allocation size must be > 0");
  void* p = ::mmap(nullptr, static_cast<size_t>(nbytes), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  TORCH_CHECK(p != MAP_FAILED, "Engram resident: mmap(", nbytes, " B) failed: ", std::strerror(errno));
  if (hugepage) {
    ::madvise(p, static_cast<size_t>(nbytes), MADV_HUGEPAGE);  // advisory (THP may be disabled)
  }
  if (!nodes.empty()) {
    unsigned long mask[16] = {0};  // up to 1024 nodes
    for (int64_t n : nodes) {
      TORCH_CHECK(n >= 0 && n < 1024, "Engram resident: NUMA node ", n, " out of range");
      mask[n / (8 * sizeof(unsigned long))] |= 1UL << (n % (8 * sizeof(unsigned long)));
    }
    long rc = ::syscall(SYS_mbind, p, static_cast<unsigned long>(nbytes), MPOL_INTERLEAVE, mask,
                        static_cast<unsigned long>(1024 + 1), 0U);
    if (rc != 0) {
      int err = errno;
      ::munmap(p, static_cast<size_t>(nbytes));
      TORCH_CHECK(false, "Engram resident: mbind(MPOL_INTERLEAVE) failed: errno ", err, " (", std::strerror(err),
                  ") -- are the nodes in this process's Mems_allowed?");
    }
  }
  const size_t len = static_cast<size_t>(nbytes);
  return torch::from_blob(
      p, {nbytes}, [len](void* q) { ::munmap(q, len); }, torch::TensorOptions().dtype(torch::kUInt8));
}

// Fill dst from `path`: for every row r of `ranges` [k, 3] int64 = (file offset, dst offset, nbytes), copy those
// bytes. Pieces of <= chunk_bytes are claimed by `threads` threads. O_DIRECT: 4 KiB-aligned reads into per-thread
// bounce buffers (bypasses the page cache / ZFS ARC), then memcpy. Returns {bytes, ns}.
std::vector<int64_t> fill_resident(torch::Tensor dst, const std::string& path, torch::Tensor ranges, int64_t threads,
                                   bool o_direct, int64_t chunk_bytes) {
  TORCH_CHECK(dst.device().is_cpu() && dst.scalar_type() == at::kByte && dst.is_contiguous(),
              "Engram resident: dst must be contiguous uint8 CPU memory");
  TORCH_CHECK(ranges.device().is_cpu() && ranges.scalar_type() == at::kLong && ranges.is_contiguous() &&
                  ranges.dim() == 2 && ranges.size(1) == 3,
              "Engram resident: ranges must be int64 [k, 3] contiguous CPU");
  TORCH_CHECK(threads >= 1 && threads <= 64, "Engram resident: threads must be in [1, 64]");
  TORCH_CHECK(chunk_bytes >= kAlign && chunk_bytes % kAlign == 0, "Engram resident: chunk_bytes must be a multiple of 4096");
  int flags = O_RDONLY | O_CLOEXEC | (o_direct ? O_DIRECT : 0);
  int fd = ::open(path.c_str(), flags);
  TORCH_CHECK(fd >= 0, "Engram resident: open(", path, o_direct ? ", O_DIRECT" : "", ") failed: ", std::strerror(errno));
  struct stat st;
  if (::fstat(fd, &st) != 0) {
    int err = errno;
    ::close(fd);
    TORCH_CHECK(false, "Engram resident: fstat(", path, ") failed: ", std::strerror(err));
  }
  const int64_t fsize = st.st_size;
  uint8_t* base = dst.data_ptr<uint8_t>();
  const int64_t dsize = dst.numel();
  const int64_t* rg = ranges.data_ptr<int64_t>();
  const int64_t k = ranges.size(0);
  // pieces: (file offset, dst offset, nbytes), each <= chunk_bytes; O_DIRECT pieces start 4 KiB-aligned in the file
  // except possibly the first piece of a range
  std::vector<int64_t> pieces;
  for (int64_t r = 0; r < k; ++r) {
    const int64_t fo = rg[3 * r], d = rg[3 * r + 1], n = rg[3 * r + 2];
    if (fo < 0 || d < 0 || n < 0 || fo + n > fsize || d + n > dsize) {
      ::close(fd);
      TORCH_CHECK(false, "Engram resident: range ", r, " (file ", fo, ", dst ", d, ", ", n, " B) outside ", path,
                  " (", fsize, " B) or dst (", dsize, " B)");
    }
    int64_t a = fo;
    while (a < fo + n) {
      int64_t b = std::min(fo + n, (a / chunk_bytes + 1) * chunk_bytes);  // chunk boundaries are file-aligned
      pieces.push_back(a);
      pieces.push_back(d + (a - fo));
      pieces.push_back(b - a);
      a = b;
    }
  }
  const int64_t np = static_cast<int64_t>(pieces.size()) / 3;
  std::atomic<int64_t> next{0};
  std::atomic<int64_t> bytes{0};
  std::mutex err_mu;
  std::string error;
  const int64_t t0 = now_ns();
  auto work = [&]() {
    void* bounce = nullptr;
    if (o_direct && ::posix_memalign(&bounce, kAlign, static_cast<size_t>(chunk_bytes + 2 * kAlign)) != 0) {
      std::lock_guard<std::mutex> g(err_mu);
      if (error.empty()) error = "posix_memalign failed";
      return;
    }
    while (true) {
      const int64_t i = next.fetch_add(1);
      if (i >= np) break;
      {
        std::lock_guard<std::mutex> g(err_mu);
        if (!error.empty()) break;
      }
      const int64_t fo = pieces[3 * i], d = pieces[3 * i + 1], n = pieces[3 * i + 2];
      int64_t ra = fo, rlen = n;
      uint8_t* out = base + d;
      if (o_direct) {
        ra = fo / kAlign * kAlign;
        rlen = (fo + n + kAlign - 1) / kAlign * kAlign - ra;
        out = static_cast<uint8_t*>(bounce);
      }
      const int64_t need = fo + n - ra;  // the aligned read may stop at EOF after the needed bytes
      int64_t got = 0;
      while (got < need) {
        ssize_t r = ::pread(fd, out + got, static_cast<size_t>(rlen - got), ra + got);
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) {
          std::ostringstream os;
          os << "pread" << (o_direct ? "(O_DIRECT)" : "") << " " << path << " offset " << ra + got << " len "
             << rlen - got << " returned " << r << " errno " << (r < 0 ? errno : EIO) << " ("
             << std::strerror(r < 0 ? errno : EIO) << ")";
          std::lock_guard<std::mutex> g(err_mu);
          if (error.empty()) error = os.str();
          break;
        }
        got += r;
      }
      if (got < need) break;
      if (o_direct) std::memcpy(base + d, out + (fo - ra), static_cast<size_t>(n));
      bytes.fetch_add(n);
    }
    if (bounce) std::free(bounce);
  };
  std::vector<std::thread> pool;
  for (int64_t t = 0; t < threads; ++t) pool.emplace_back(work);
  for (auto& t : pool) t.join();
  if (!o_direct) {
    for (int64_t r = 0; r < k; ++r) {  // the resident copy is the only one kept (no-op on ZFS: ARC is not the page cache)
      ::posix_fadvise(fd, rg[3 * r], rg[3 * r + 2], POSIX_FADV_DONTNEED);
    }
  }
  ::close(fd);
  TORCH_CHECK(error.empty(), "Engram resident load failed: ", error);
  return {bytes.load(), now_ns() - t0};
}

// Step gather from the resident tables: for i in [0, n): row_bytes from rows[tab[i]] entry sidx[i] -> dst + i *
// dst_stride, then extra_bytes from extras[tab[i]] entry sidx[i] right after it (extras empty: no extra bytes).
// Same output as RowReader.submit with offsets = the row's file offset and (xtab, xidx) = (tab, sidx).
int64_t gather_resident(const std::vector<torch::Tensor>& rows, const std::vector<torch::Tensor>& extras,
                        torch::Tensor tab, torch::Tensor sidx, torch::Tensor dst, int64_t dst_stride,
                        int64_t row_bytes, int64_t extra_bytes) {
  const int64_t n = sidx.numel();
  TORCH_CHECK(tab.device().is_cpu() && tab.scalar_type() == at::kInt && tab.is_contiguous() && tab.numel() == n,
              "Engram resident: tab must be int32 [n] contiguous CPU");
  TORCH_CHECK(sidx.device().is_cpu() && sidx.scalar_type() == at::kLong && sidx.is_contiguous(),
              "Engram resident: sidx must be int64 [n] contiguous CPU");
  TORCH_CHECK(dst.device().is_cpu() && dst.scalar_type() == at::kByte && dst.is_contiguous(),
              "Engram resident: dst must be contiguous uint8 CPU memory");
  TORCH_CHECK(row_bytes > 0 && extra_bytes >= 0 && dst_stride >= row_bytes + (extras.empty() ? 0 : extra_bytes),
              "Engram resident: dst_stride ", dst_stride, " too small for row ", row_bytes, " + extra ", extra_bytes);
  TORCH_CHECK(dst.numel() >= n * dst_stride, "Engram resident: dst holds ", dst.numel(), " bytes, need ", n * dst_stride);
  TORCH_CHECK(extras.empty() || extras.size() == rows.size(), "Engram resident: one extra table per row table");
  std::vector<const uint8_t*> rb, xb;
  std::vector<int64_t> entries;
  for (size_t t = 0; t < rows.size(); ++t) {
    TORCH_CHECK(rows[t].device().is_cpu() && rows[t].scalar_type() == at::kByte && rows[t].is_contiguous() &&
                    rows[t].numel() % row_bytes == 0,
                "Engram resident: row table ", t, " must be contiguous uint8 CPU, a multiple of row_bytes");
    rb.push_back(rows[t].data_ptr<uint8_t>());
    entries.push_back(rows[t].numel() / row_bytes);
    if (!extras.empty()) {
      TORCH_CHECK(extras[t].device().is_cpu() && extras[t].scalar_type() == at::kByte && extras[t].is_contiguous() &&
                      extras[t].numel() == entries.back() * extra_bytes,
                  "Engram resident: extra table ", t, " holds ", extras[t].numel(), " B, expected ",
                  entries.back() * extra_bytes);
      xb.push_back(extras[t].data_ptr<uint8_t>());
    }
  }
  const int32_t* tp = tab.data_ptr<int32_t>();
  const int64_t* sp = sidx.data_ptr<int64_t>();
  for (int64_t i = 0; i < n; ++i) {
    TORCH_CHECK(tp[i] >= 0 && tp[i] < static_cast<int32_t>(rb.size()), "Engram resident: table id ", tp[i],
                " not loaded");
    TORCH_CHECK(sp[i] >= 0 && sp[i] < entries[tp[i]], "Engram resident: row index ", sp[i], " outside table ", tp[i],
                " (", entries[tp[i]], " rows)");
  }
  uint8_t* out = dst.data_ptr<uint8_t>();
  const int64_t t0 = now_ns();
  constexpr int64_t kAhead = 8;
  for (int64_t i = 0; i < n; ++i) {
    if (i + kAhead < n) {  // random 256 B rows in tens of GB: overlap the DRAM misses
      const uint8_t* q = rb[tp[i + kAhead]] + sp[i + kAhead] * row_bytes;
      for (int64_t c = 0; c < row_bytes; c += 64) __builtin_prefetch(q + c, 0, 0);
    }
    std::memcpy(out + i * dst_stride, rb[tp[i]] + sp[i] * row_bytes, static_cast<size_t>(row_bytes));
    if (!xb.empty()) {
      std::memcpy(out + i * dst_stride + row_bytes, xb[tp[i]] + sp[i] * extra_bytes, static_cast<size_t>(extra_bytes));
    }
  }
  return now_ns() - t0;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  namespace py = pybind11;
  m.def("alloc_resident", &alloc_resident, py::arg("nbytes"), py::arg("nodes"), py::arg("hugepage") = true);
  m.def("fill_resident", &fill_resident, py::arg("dst"), py::arg("path"), py::arg("ranges"), py::arg("threads"),
        py::arg("o_direct"), py::arg("chunk_bytes"), py::call_guard<py::gil_scoped_release>());
  m.def("gather_resident", &gather_resident, py::arg("rows"), py::arg("extras"), py::arg("tab"), py::arg("sidx"),
        py::arg("dst"), py::arg("dst_stride"), py::arg("row_bytes"), py::arg("extra_bytes"),
        py::call_guard<py::gil_scoped_release>());
}
