// This enum specifies device type, rather than actual devices.
// It is used to determine whether to use cuda or cpu
// implementations of various functions.
#ifndef DALI_ARRAY_MEMORY_DEVICE_H
#define DALI_ARRAY_MEMORY_DEVICE_H

#include <map>
#include <vector>

#include "dali/config.h"

namespace memory {
    #ifdef DALI_USE_CUDA
        const int MAX_GPU_DEVICES = 16;
    #else
        const int MAX_GPU_DEVICES = 0;
    #endif

#ifdef DALI_USE_CUDA
    enum DeviceT {
        DEVICE_T_ERROR=0,
        DEVICE_T_FAKE=1,
        DEVICE_T_CPU=2,
        DEVICE_T_GPU=3,
    };
#else
    enum DeviceT {
        DEVICE_T_ERROR=0,
        DEVICE_T_FAKE=1,
        DEVICE_T_CPU=2,
    };
#endif

    // allows conversion between a device enum
    // and its printable name (e.g. cpu, gpu)
    extern std::map<DeviceT, std::string> device_type_to_name;

    class Device {
      private:
        DeviceT mType;
        // ignored for cpu:
        int mNumber;

        Device(DeviceT type, int number);
      public:
        Device();

        DeviceT type() const;
        int number() const;
        std::string description(bool real_gpu_name=false) const;

        // fake devices
        bool is_fake() const;
        static Device fake(int number);

        // cpu devices
        bool is_cpu() const;
        static Device cpu();
        static Device device_of_doom();
        // static functions
        static std::vector<memory::Device> installed_devices();
#ifdef DALI_USE_CUDA
        // gpu devices
        void set_cuda_device() const;
        bool is_gpu() const;
        static Device gpu(int number);
        static int num_gpus();
#endif
    };

    bool operator==(const Device& a, const Device& b);
    bool operator!=(const Device& a, const Device& b);

    struct DevicePtr {
        Device device;
        void* ptr;

        DevicePtr(Device _device, void* _ptr);
    };


};  // namespace memory



#endif
