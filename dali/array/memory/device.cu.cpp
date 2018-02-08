#include "dali/array/memory/device.h"

#include "dali/config.h"
#include "dali/utils/assert2.h"
#include "dali/utils/make_message.h"

namespace memory {
    std::map<DeviceT, std::string> device_type_to_name = {
        {DEVICE_T_CPU,    "cpu"},
        {DEVICE_T_GPU,    "gpu"},
    };

    Device::Device() :
            mType(DEVICE_T_ERROR), mNumber(-1) {
    }

    Device::Device(DeviceT _type, int _number) :
            mType(_type), mNumber(_number) {
    }

    DeviceT Device::type() const {
        return mType;
    }
    int Device::number() const {
        return mNumber;
    }

    std::string Device::description(bool real_gpu_name) const {
        if (is_cpu()) {
            return "cpu";
        } else if(is_gpu()) {
            std::string real_name;
            if (real_gpu_name) {
                real_name = utils::make_message(" (", gpu_name(), ")");
            }
            return utils::make_message("gpu", mNumber, real_name);
        } else if(is_fake()) {
            return utils::make_message("fake_device", mNumber);
        } else if(mType == DEVICE_T_ERROR) {
            return "device_of_doom";
        } else {
            ASSERT2(false, "Device::description: unknown device type stored in Device class.");
            return "";
        }
    }


    bool Device::is_cpu() const {
        return mType == DEVICE_T_CPU;
    }

    Device Device::cpu() {
        return Device(DEVICE_T_CPU, -1);
    }

    bool Device::is_fake() const {
        return mType == DEVICE_T_FAKE;
    }

    bool Device::is_error() const {
        return mType == DEVICE_T_ERROR;
    }

    Device Device::fake(int number) {
        ASSERT2(debug::enable_fake_devices,
                "To create a fake device, you must first set "
                "memory::debug::enable_fake_devices to true.");
        return Device(DEVICE_T_FAKE, number);
    }

    Device Device::device_of_doom() {
        return Device(DEVICE_T_ERROR, -1);
    }


    std::vector<memory::Device> Device::installed_devices() {
        std::vector<memory::Device> result;
        result.push_back(Device::cpu()); // the day this line will be iffed guared
                                         // I will know the future is here.
        for (int i=0; i < num_gpus(); ++i) {
            result.push_back(Device::gpu(i));
        }
        return result;
    }

    void Device::set_cuda_device() const {
#ifdef DALI_USE_CUDA
        ASSERT2(is_gpu(), "set_cuda_device must only be called for GPU devices.");
        cudaSetDevice(mNumber);
#endif
    }

    bool Device::is_gpu() const {
        return mType == DEVICE_T_GPU;
    }

    Device Device::gpu(int number) {
        ASSERT2(0 <= number && number < DALI_MAX_GPU_DEVICES, utils::make_message(
            "GPU number must be between 0 and ", DALI_MAX_GPU_DEVICES - 1, "."));
        return Device(DEVICE_T_GPU, number);
    }

    int Device::num_gpus() {
#ifdef DALI_USE_CUDA
        int ndevices;
        cudaGetDeviceCount(&ndevices);
        return ndevices;
#else
        return 0;
#endif  // DALI_USE_CUDA
    }

    std::string Device::gpu_name() const {
        ASSERT2(is_gpu(), "gpu_name must only be called for GPU devices.");
#ifdef DALI_USE_CUDA
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, mNumber);
        return std::string(props.name);
#else
        return "GPU";
#endif
    }

    bool operator==(const Device& a, const Device& b) {
        return a.type() == b.type() && a.number() == b.number();
    }
    bool operator!=(const Device& a, const Device& b) {
        return !(a==b);
    }

    DevicePtr::DevicePtr(Device _device, void* _ptr) : device(_device), ptr(_ptr) {}

    namespace debug {
        bool enable_fake_devices = false;
    }
}  // namespace memory
