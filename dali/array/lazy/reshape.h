#ifndef DALI_ARRAY_LAZY_RESHAPE_H
#define DALI_ARRAY_LAZY_RESHAPE_H

#include "dali/array/function/expression.h"

template<typename SrcExp, typename IndexExp>
struct LazyTake;

template<typename SrcExp, typename IndexExp>
struct LazyTakeFromRows;

namespace lazy {
    template<typename SrcExp, typename IndexExp>
    LazyTake<SrcExp, IndexExp> take(const SrcExp& source, const IndexExp& indices);

    template<typename SrcExp, typename IndexExp>
    LazyTakeFromRows<SrcExp, IndexExp> take_from_rows(const SrcExp& source, const IndexExp& indices);
}  // namespace lazy

#include "dali/array/lazy/reshape-impl.h"

#endif
