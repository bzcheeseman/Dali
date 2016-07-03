#ifndef DALI_ARRAY_LAZY_CIRCULAR_CONVOLUTION_H
#define DALI_ARRAY_LAZY_CIRCULAR_CONVOLUTION_H

template<typename ContentExp, typename ShiftExp>
struct LazyCircularConvolution;

namespace lazy {
    template<typename ContentExp, typename ShiftExp>
    LazyCircularConvolution<ContentExp, ShiftExp> circular_convolution(
        const ContentExp& content, const ShiftExp& shift
    );
}  // namespace lazy

#include "dali/array/lazy/circular_convolution-impl.h"

#endif  // DALI_ARRAY_LAZY_CIRCULAR_CONVOLUTION_H
