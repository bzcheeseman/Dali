#include "dali/array/memory/device.h"

#include "dali/config.h"
#include "dali/utils/assert2.h"
#include "dali/utils/print_utils.h"

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

    Device::Device() :
            type(DEVICE_T_ERROR), number(-1) {
    }

    std::string Device::description() {
        if (is_cpu()) {
            return "cpu";
        } else if(is_gpu()) {
            return utils::MS() << "gpu" << number;
        } else if(is_fake()) {
            return utils::MS() << "fake_device" << number;
        } else if(type==DEVICE_T_ERROR) {
            return "device_of_doom";
        } else {
            ASSERT2(false, "Device::description: unknown device type stored in Device class.");
            return "";
        }
    }

    Device::Device(DeviceT _type, int _number) :
            type(_type), number(_number) {
    }

    bool Device::is_cpu() const {
        return type == DEVICE_T_CPU;
    }

    Device Device::cpu() {
        return Device(DEVICE_T_CPU, -1);
    }

    bool Device::is_fake() const {
        return type == DEVICE_T_FAKE;
    }

    Device Device::fake(int number) {
        return Device(DEVICE_T_FAKE, number);
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
        ASSERT2(0 <= number && number < MAX_GPU_DEVICES,
                utils::MS() << "GPU number must be between 0 and " << MAX_GPU_DEVICES - 1 << ".");
        return Device(DEVICE_T_GPU, number);
    }
#endif


    bool operator==(const Device& a, const Device& b) {
        return a.type == b.type && a.number == b.number;
    }
    bool operator!=(const Device& a, const Device& b) {
        return !(a==b);
    }

    DevicePtr::DevicePtr(Device _device, void* _ptr) : device(_device), ptr(_ptr) {}

}  // namespace memory
