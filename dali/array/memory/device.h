// This enum specifies device type, rather than actual devices.
// It is used to determine whether to use cuda or cpu
// implementations of various functions.
#ifndef DALI_ARRAY_MEMORY_DEVICE_H
#define DALI_ARRAY_MEMORY_DEVICE_H

#include <map>
#include "dali/config.h"

namespace memory {

#ifdef DALI_USE_CUDA
    enum DeviceT {
        DEVICE_T_ERROR=0,
        DEVICE_T_CPU=1,
        DEVICE_T_GPU=2,
    };
#else
    enum DeviceT {
        DEVICE_T_ERROR=0,
        DEVICE_T_CPU=1,
    };
#endif

    // allows conversion between a device enum
    // and its printable name (e.g. cpu, gpu)
    extern std::map<DeviceT, std::string> device_type_to_name;

    struct Device {
        const DeviceT type;
        // ignored for cpu:
        const int number;

        bool is_cpu() const;
        static Device cpu();
        static Device device_of_doom();
#ifdef DALI_USE_CUDA
        void set_cuda_device() const;
        bool is_gpu() const;
        static Device gpu(int number);
#endif

        private:
            Device(DeviceT _type, int _number=0);
    };

    struct DevicePtr {
        Device device;
        void* ptr;

        DevicePtr(Device _device, void* _ptr);
    };


};  // namespace memory



#endif
