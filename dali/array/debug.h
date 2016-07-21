#ifndef DALI_ARRAY_DEBUG_H
#define DALI_ARRAY_DEBUG_H

#include "dali/utils/debug_callback.h"

class Array;

namespace debug {
    extern Observation<std::string, int> dali_function_start;
    extern Observation<std::string, int> dali_function_end;
    extern Observation<Array>            lazy_evaluation_callback;
    extern Observation<Array>            array_as_contiguous;
}  // namespace debug

#endif // DALI_ARRAY_DEBUG_H
