#ifndef DALI_ARRAY_LAZY_RESHAPE_H
#define DALI_ARRAY_LAZY_RESHAPE_H

#include "dali/array/function/expression.h"

template<typename SrcExp, typename IndexExp>
struct LazyTake;

namespace lazy {
    template<typename SrcExp, typename IndexExp>
    LazyTake<SrcExp, IndexExp> take(const SrcExp& source, const IndexExp& indices);
}  // namespace lazy

#include "dali/array/lazy/reshape-impl.h"

#endif
