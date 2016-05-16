#ifndef DALI_ARRAY_LAZY_REDUCERS_H
#define DALI_ARRAY_LAZY_REDUCERS_H

template<class Functor, typename ExprT>
struct LazyAllReducer;

template<class Functor, typename ExprT, bool return_indices>
struct LazyAxisReducer;

namespace myops {
    struct sum_all;
    struct sum_axis;
}

namespace mshadow {
	namespace red {
		struct sum;
        struct maximum;
        struct minimum;
	}
};

namespace lazy {
    template<typename ExprT>
    LazyAllReducer<myops::sum_all, ExprT> sum_all(const Exp<ExprT>& expr);

    template<typename ExprT>
    LazyAxisReducer<mshadow::red::sum, ExprT, false> sum_axis(const Exp<ExprT>& expr, const int& axis, bool keepdims=false);

    template<typename ExprT>
    LazyAxisReducer<mshadow::red::maximum, ExprT, true> argmax_axis(const Exp<ExprT>& expr, const int& axis, bool keepdims=false);

    template<typename ExprT>
    LazyAxisReducer<mshadow::red::maximum, ExprT, false> max_axis(const Exp<ExprT>& expr, const int& axis, bool keepdims=false);

    template<typename ExprT>
    LazyAxisReducer<mshadow::red::minimum, ExprT, true> argmin_axis(const Exp<ExprT>& expr, const int& axis, bool keepdims=false);

    template<typename ExprT>
    LazyAxisReducer<mshadow::red::minimum, ExprT, false> min_axis(const Exp<ExprT>& expr, const int& axis, bool keepdims=false);

}  // namespace lazy

#include "dali/array/lazy/reducers-impl.h"

#endif
