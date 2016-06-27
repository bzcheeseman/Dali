#ifndef DALI_RUNTIME_CONFIG
#define DALI_RUNTIME_CONFIG

#include "dali/array/memory/device.h"


extern bool use_cudnn;

namespace memory {
    extern memory::Device default_preferred_device;

    class WithDevicePreference {
        private:
            // value of backprop before object go activated.
            const Device old_value;

            WithDevicePreference(const WithDevicePreference&) = delete;
            WithDevicePreference& operator =(WithDevicePreference const &) = delete;

        public:
            WithDevicePreference() = delete;
            explicit WithDevicePreference(Device device);
            ~WithDevicePreference();
    };
}
#endif
