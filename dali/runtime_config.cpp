#include "dali/runtime_config.h"

#include "dali/config.h"
#include "dali/array/memory/device.h"

using memory::Device;

#ifdef DALI_USE_CUDA
    Device default_preferred_device = Device::gpu(0);
#else
    Device default_preferred_device = Device::cpu();
#endif
