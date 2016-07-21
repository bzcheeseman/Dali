#include "debug.h"

#include "dali/array/array.h"

namespace debug {
    DebugCallback<std::string, int> dali_function_start;
    DebugCallback<std::string, int> dali_function_end;
    DebugCallback<Array> lazy_evaluation_callback;
    DebugCallback<Array> array_as_contiguous;
}  // namespace debug
