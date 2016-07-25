#include "debug.h"

#include "dali/array/array.h"

namespace debug {
    Observation<Array> lazy_evaluation_callback;
    Observation<Array> array_as_contiguous;
}  // namespace debug
