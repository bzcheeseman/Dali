#include "debug.h"

#include "dali/array/array.h"

namespace debug {
    Observation<std::string, int> dali_function_start;
    Observation<std::string, int> dali_function_end;
    Observation<Array> lazy_evaluation_callback;
    Observation<Array> array_as_contiguous;
}  // namespace debug
