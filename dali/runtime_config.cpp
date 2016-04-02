#include "dali/runtime_config.h"

#include "dali/config.h"

using namespace memory_ops;

#ifdef DALI_USE_CUDA
    Device preferred_device = DEVICE_GPU;
#else
    Device preferred_device = DEVICE_CPU;
#endif
