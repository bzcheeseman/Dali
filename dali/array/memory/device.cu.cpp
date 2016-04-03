#include "dali/array/memory/device.h"

#include "dali/utils/assert2.h"

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


    DevicePtr::DevicePtr(Device _device, void* _ptr) : device(_device), ptr(_ptr) {}

    bool Device::is_cpu() const {
        return type == DEVICE_T_CPU;
    }

    Device Device::cpu() {
        return Device(DEVICE_T_CPU, -1);
    }

    Device Device::device_of_doom() {
        return Device(DEVICE_T_ERROR, -1);
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
        return Device(DEVICE_T_GPU, number);
    }
#endif

    Device::Device(DeviceT _type, int _number) : type(_type), number(_number) {}
}  // namespace memory
