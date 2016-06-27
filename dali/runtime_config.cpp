#include "dali/runtime_config.h"

#include "dali/config.h"
#include "dali/array/memory/device.h"

#ifdef DALI_USE_CUDNN
    bool use_cudnn = true;
#else
    bool use_cudnn = false;
#endif

namespace memory {
    #ifdef DALI_USE_CUDA
        Device default_preferred_device = Device::gpu(0);
    #else
        Device default_preferred_device = Device::cpu();
    #endif

    WithDevicePreference::WithDevicePreference(Device new_device) :
            old_value(default_preferred_device) {
        default_preferred_device = new_device;
    }

    WithDevicePreference::~WithDevicePreference() {
        default_preferred_device = old_value;
    }
}
