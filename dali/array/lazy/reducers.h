#ifndef DALI_ARRAY_LAZY_REDUCERS_H
#define DALI_ARRAY_LAZY_REDUCERS_H

template<class Functor, typename ExprT>
struct LazyAllReducer;

template<class Functor, typename ExprT>
struct LazyAxisReducer;

namespace myops {
    struct sum_all;
    struct sum_axis;
}

namespace mshadow {
	namespace red {
		struct sum;
	}
};

namespace lazy {
    template<typename ExprT>
    LazyAllReducer<myops::sum_all, ExprT> sum_all(const Exp<ExprT>& expr);

    template<typename ExprT>
    LazyAxisReducer<mshadow::red::sum, ExprT> sum_axis(const Exp<ExprT>& expr, const int& reduce_axis);
}  // namespace lazy

#include "dali/array/lazy/reducers-impl.h"

#endif
