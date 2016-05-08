#ifndef DALI_ARRAY_LAZY_REDUCERS_H
#define DALI_ARRAY_LAZY_REDUCERS_H

template<class Functor, typename ExprT>
struct LazyAllReducer;

namespace myops {
    struct sum_all;
}

namespace lazy {
    template<typename ExprT>
    LazyAllReducer<myops::sum_all, ExprT> sum_all(const Exp<ExprT>& expr);
}  // namespace lazy

#include "dali/array/lazy/reducers-impl.h"

#endif
