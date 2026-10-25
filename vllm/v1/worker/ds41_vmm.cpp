// CUDA virtual memory for DS41 KV cache.
// Virtual address covers the full pool. Physical  pages are created on map
// and returned to the driver on unmap. Tested on this V100: granularity is
// 2 MiB, cuMemAddressReserve does not consume GPU memory, cuMemCreate does,
// and cuMemUnmap + cuMemRelease gives it back.

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

void check_cu(CUresult result, const char* what) {
    if (result == CUDA_SUCCESS) {
        return;
    }
    const char* name = nullptr;
    const char* msg = nullptr;
    cuGetErrorName(result, &name);
    cuGetErrorString(result, &msg);
    throw std::runtime_error(std::string(what) + ": " + (name ? name : "?") +
                             " " + (msg ? msg : ""));
}

class Pool {
public:
    Pool(int64_t nbytes, int device_index)
        : nbytes_(nbytes), device_index_(device_index) {
        if (nbytes <= 0) {
            throw std::runtime_error("ds41_vmm: nbytes must be positive");
        }
        check_cu(cuInit(0), "cuInit");
        c10::cuda::CUDAGuard guard(device_index_);
        check_cu(cuCtxGetDevice(&device_), "cuCtxGetDevice");

        prop_ = {};
        prop_.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop_.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop_.location.id = device_;

        check_cu(cuMemGetAllocationGranularity(
                     &gran_, &prop_, CU_MEM_ALLOC_GRANULARITY_MINIMUM),
                 "cuMemGetAllocationGranularity");
        if (gran_ == 0) {
            throw std::runtime_error("ds41_vmm: granularity is 0");
        }

        nchunks_ = static_cast<size_t>((static_cast<uint64_t>(nbytes_) + gran_ - 1) / gran_);
        reserve_ = nchunks_ * gran_;
        check_cu(cuMemAddressReserve(&va_, reserve_, gran_, 0, 0),
                 "cuMemAddressReserve");

        handles_.assign(nchunks_, 0);
        mapped_.assign(nchunks_, 0);
    }

    Pool(const Pool&) = delete;
    Pool& operator=(const Pool&) = delete;

    ~Pool() {
        // If a tensor still wraps this VA, leak the reservation. The driver
        // reclaims it when the process exits. Freeing it under a live
        // from_blob storage would fault the worker at shutdown.
        bool sole = !tensor_.defined() || tensor_.use_count() <= 1;
        tensor_ = at::Tensor();
        if (!sole || va_ == 0) {
            return;
        }
        CUcontext ctx = nullptr;
        if (cuCtxGetCurrent(&ctx) != CUDA_SUCCESS || ctx == nullptr) {
            return;
        }
        for (size_t i = 0; i < mapped_.size(); ++i) {
            if (!mapped_[i]) {
                continue;
            }
            cuMemUnmap(va_ + i * gran_, gran_);
            cuMemRelease(handles_[i]);
            mapped_[i] = 0;
        }
        cuMemAddressFree(va_, reserve_);
        va_ = 0;
    }

    int64_t granularity() const { return static_cast<int64_t>(gran_); }
    int64_t nchunks() const { return static_cast<int64_t>(nchunks_); }
    int64_t nbytes() const { return nbytes_; }
    int64_t mapped_bytes() const {
        int64_t n = 0;
        for (char bit : mapped_) {
            n += bit ? static_cast<int64_t>(gran_) : 0;
        }
        return n;
    }

    void map_chunk(int64_t index, uint64_t stream_bits) {
        if (index < 0 || static_cast<size_t>(index) >= nchunks_) {
            throw std::runtime_error("ds41_vmm: chunk index out of range");
        }
        size_t i = static_cast<size_t>(index);
        if (mapped_[i]) {
            return;
        }
        c10::cuda::CUDAGuard guard(device_index_);
        CUmemGenericAllocationHandle handle = 0;
        check_cu(cuMemCreate(&handle, gran_, &prop_, 0), "cuMemCreate");
        CUdeviceptr addr = va_ + i * gran_;
        CUresult mapped = cuMemMap(addr, gran_, 0, handle, 0);
        if (mapped != CUDA_SUCCESS) {
            cuMemRelease(handle);
            check_cu(mapped, "cuMemMap");
        }
        CUmemAccessDesc access = {};
        access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        access.location.id = device_;
        access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        CUresult access_rc = cuMemSetAccess(addr, gran_, &access, 1);
        if (access_rc != CUDA_SUCCESS) {
            cuMemUnmap(addr, gran_);
            cuMemRelease(handle);
            check_cu(access_rc, "cuMemSetAccess");
        }
        // Fresh physical pages are garbage. State-cache / Mamba must not see them.
        CUstream stream = reinterpret_cast<CUstream>(stream_bits);
        check_cu(cuMemsetD8Async(addr, 0, gran_, stream), "cuMemsetD8Async");
        handles_[i] = handle;
        mapped_[i] = 1;
    }

    void unmap_chunk(int64_t index) {
        if (index < 0 || static_cast<size_t>(index) >= nchunks_) {
            throw std::runtime_error("ds41_vmm: chunk index out of range");
        }
        size_t i = static_cast<size_t>(index);
        if (!mapped_[i]) {
            return;
        }
        CUdeviceptr addr = va_ + i * gran_;
        check_cu(cuMemUnmap(addr, gran_), "cuMemUnmap");
        check_cu(cuMemRelease(handles_[i]), "cuMemRelease");
        handles_[i] = 0;
        mapped_[i] = 0;
    }

    at::Tensor view() {
        c10::cuda::CUDAGuard guard(device_index_);
        if (!tensor_.defined()) {
            // from_blob rejects a reserved VA whose first page is not mapped:
            // cudaPointerGetAttributes reports it as unregistered host memory.
            if (nchunks_ > 0 && !mapped_[0]) {
                map_chunk(
                    0,
                    reinterpret_cast<uint64_t>(
                        c10::cuda::getCurrentCUDAStream().stream()));
            }
            auto opts = torch::TensorOptions()
                            .dtype(torch::kInt8)
                            .device(torch::kCUDA, device_index_);
            tensor_ = torch::from_blob(
                reinterpret_cast<void*>(va_),
                {nbytes_},
                [](void*) {},
                opts);
        }
        return tensor_;
    }

private:
    int64_t nbytes_;
    int device_index_;
    CUdevice device_ = 0;
    CUmemAllocationProp prop_{};
    size_t gran_ = 0;
    size_t nchunks_ = 0;
    size_t reserve_ = 0;
    CUdeviceptr va_ = 0;
    std::vector<CUmemGenericAllocationHandle> handles_;
    std::vector<char> mapped_;
    at::Tensor tensor_;
};

std::pair<int64_t, int64_t> mem_info() {
    check_cu(cuInit(0), "cuInit");
    CUcontext ctx = nullptr;
    CUresult current = cuCtxGetCurrent(&ctx);
    if (current != CUDA_SUCCESS || ctx == nullptr) {
        // Make the runtime primary context current. cuMemGetInfo on a bare
        // driver call has no context until PyTorch (or this) has touched one.
        cudaFree(0);
    }
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    check_cu(cuMemGetInfo(&free_bytes, &total_bytes), "cuMemGetInfo");
    return {static_cast<int64_t>(free_bytes), static_cast<int64_t>(total_bytes)};
}

}  // namespace

PYBIND11_MODULE(ds41_vmm, m) {
    pybind11::class_<Pool>(m, "Pool")
        .def(pybind11::init<int64_t, int>())
        .def("granularity", &Pool::granularity)
        .def("nchunks", &Pool::nchunks)
        .def("nbytes", &Pool::nbytes)
        .def("mapped_bytes", &Pool::mapped_bytes)
        .def("map_chunk", &Pool::map_chunk)
        .def("unmap_chunk", &Pool::unmap_chunk)
        .def("view", &Pool::view);
    m.def("mem_info", &mem_info, "Driver free and total bytes on the current device");
}
