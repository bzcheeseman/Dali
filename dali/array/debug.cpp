#include "debug.h"

#include "dali/array/array.h"

namespace debug {
    DebugCallback<bool>      dali_function_computed;
    DebugCallback<Array> lazy_evaluation_callback;
    DebugCallback<Array> array_as_contiguous;
}  // namespace debug
