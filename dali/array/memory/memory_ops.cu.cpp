#include "dali/array/memory/memory_ops.h"

#include <cstdlib>

#include "dali/array/memory/device.h"
#include "dali/utils/assert2.h"
#include "dali/utils/make_message.h"

using utils::assert2;

#ifdef DALI_USE_CUDA
#define DALI_CUDA_CALL(func)                                       \
  {                                                                \
    cudaError_t e = (func);                                        \
    ASSERT2(e != cudaErrorCudartUnloading, cudaGetErrorString(e)); \
    ASSERT2(e == cudaSuccess, utils::make_message("CUDA: ", cudaGetErrorString(e)));\
  }
#endif

namespace memory {
    enum PacketArch {kPlain, kSSE2};

    #if DALI_USE_SSE
    #define DALI_DEFAULT_PACKET  memory::kSSE2
    #else
    #define DALI_DEFAULT_PACKET  memory::kPlain
    #endif

    template<PacketArch Arch>
    struct AlignBytes {static const int value = 4;};

    inline void* AlignedMallocPitch(size_t *out_pitch,
                                    size_t lspace,
                                    size_t num_line) {
        const int bits = AlignBytes<DALI_DEFAULT_PACKET>::value;
        const int mask = (1 << bits) - 1;

        size_t pitch = ((lspace + mask) >> bits) << bits;
        *out_pitch = pitch;
#ifdef _MSC_VER
        void *data = _aligned_malloc(pitch * num_line, 1 << bits);
#else
        void *data;
        int posix_memalign_result = posix_memalign(&data, 1 << bits, pitch * num_line);
        ASSERT2(posix_memalign_result == 0, "AlignedMallocPitch failed.");
#endif
        ASSERT2(data != NULL, "AlignedMallocPitch failed.");
        return data;
    }

    inline void AlignedFree(void *ptr) {
#ifdef _MSC_VER
        _aligned_free(ptr);
#else
        std::free(ptr);
#endif
    }

    DevicePtr allocate(Device device, int amount, int inner_dimension) {
        size_t pitch;
        if (device.is_cpu()) {
            auto data = AlignedMallocPitch(&pitch, amount, 1);
            return DevicePtr(device, data);
        }
#ifdef DALI_USE_CUDA
        else if (device.is_gpu()) {
            device.set_cuda_device();
            void* data;
            DALI_CUDA_CALL(cudaMallocPitch(&data, &pitch, amount, 1));
            return DevicePtr(device, data);
        }
#endif
        else {
            ASSERT2(false, "Wrong device passed to Device enum.");
            return DevicePtr(Device::device_of_doom(), (void*)NULL);
        }
    }

    void free(DevicePtr dev_ptr, int amount, int inner_dimension) {
        if (dev_ptr.device.is_cpu()) {
            AlignedFree(dev_ptr.ptr);
        }
#ifdef DALI_USE_CUDA
        else if (dev_ptr.device.is_gpu()) {
            dev_ptr.device.set_cuda_device();
            DALI_CUDA_CALL(cudaFree(dev_ptr.ptr));
        }
#endif
        else {
            ASSERT2(false, "Wrong device type passed to Device enum");
        }
    }

    void clear(DevicePtr dev_ptr, int amount, int inner_dimension) {
        if (dev_ptr.device.is_cpu()) {
            memset(dev_ptr.ptr, 0, amount);
        }
#ifdef DALI_USE_CUDA
        else if (dev_ptr.device.is_gpu()) {
            dev_ptr.device.set_cuda_device();
            DALI_CUDA_CALL(cudaMemset(dev_ptr.ptr, 0, amount));
        }
#endif
        else {
            ASSERT2(false, "Wrong device passed to Device enum.");
        }
    }

    void copy(DevicePtr dest, DevicePtr source, int amount, int inner_dimension) {
        if (dest.device.is_cpu() && source.device.is_cpu()) {
            memcpy(dest.ptr, source.ptr, amount);
        }
#ifdef DALI_USE_CUDA
        else if (dest.device.is_cpu() && source.device.is_gpu()) {
            source.device.set_cuda_device();
            cudaMemcpy(dest.ptr, source.ptr, amount, cudaMemcpyDeviceToHost);
        } else if (dest.device.is_gpu() && source.device.is_cpu()) {
            dest.device.set_cuda_device();
            cudaMemcpy(dest.ptr, source.ptr, amount, cudaMemcpyHostToDevice);
        } else if (dest.device.is_gpu() && source.device.is_gpu()) {
            dest.device.set_cuda_device();
            ASSERT2(dest.device.number() == source.device.number(), utils::make_message(
                "GPU -> GPU memory movement not supported yet (device ",
                source.device.number(), " to device ", dest.device.number(), ")."));
            cudaMemcpy(dest.ptr, source.ptr, amount, cudaMemcpyDeviceToDevice);
        }
#endif
        else {
            ASSERT2(false, "Wrong device passed to Device enum.");
        }
    }
#ifdef DALI_USE_CUDA
        size_t cuda_available_memory() {
            size_t free_memory, total_memory;
            cudaMemGetInfo(&free_memory, &total_memory);
            return free_memory;
        }
#endif
}
