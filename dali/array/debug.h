#ifndef DALI_ARRAY_DEBUG_H
#define DALI_ARRAY_DEBUG_H

#include "dali/utils/observer.h"

class Array;


namespace debug {
    extern Observation<Array>            lazy_evaluation_callback;
    extern Observation<Array>            array_as_contiguous;
}  // namespace debug

#endif // DALI_ARRAY_DEBUG_H
