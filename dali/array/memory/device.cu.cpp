#include "dali/array/memory/device.h"

#include "dali/config.h"
#include "dali/utils/assert2.h"
#include "dali/utils/core_utils.h"

namespace memory {
#ifdef DALI_USE_CUDA
    std::map<DeviceT, std::string> device_type_to_name = {
        {DEVICE_T_CPU,    "cpu"},
        {DEVICE_T_GPU,    "gpu"},
    };
#else
    std::map<DeviceT, std::string> device_type_to_name = {
        {DEVICE_T_CPU,    "cpu"},
    };
#endif



    bool Device::is_cpu() const {
        return type == DEVICE_T_CPU;
    }

    Device Device::cpu() {
        return Device{DEVICE_T_CPU, -1};
    }

    Device Device::device_of_doom() {
        return Device{DEVICE_T_ERROR, -1};
    }



#ifdef DALI_USE_CUDA
    void Device::set_cuda_device() const {
        ASSERT2(is_gpu(), "set_cuda_device must only be called for GPU devices.");
        cudaSetDevice(number);
    }

    bool Device::is_gpu() const {
        return type == DEVICE_T_GPU;
    }

    Device Device::gpu(int number) {
        ASSERT2(0 <= number && number < MAX_GPU_DEVICES,
                utils::MS() << "GPU number must be between 0 and " << MAX_GPU_DEVICES - 1 << ".");
        return Device{DEVICE_T_GPU, number};
    }
#endif


    DevicePtr::DevicePtr(Device _device, void* _ptr) : device(_device), ptr(_ptr) {}

}  // namespace memory
