#ifndef DALI_ARRAY_LAZY_REDUCERS_H
#define DALI_ARRAY_LAZY_REDUCERS_H

template<class Functor, typename ExprT>
struct LazyAllReducer;

template<class Functor, typename ExprT, bool return_indices>
struct LazyAxisReducer;

namespace mshadow {
	namespace red {
		struct sum;
        struct maximum;
        struct minimum;
	}
};

namespace lazy {
    template<typename ExprT>
    LazyAllReducer<mshadow::red::sum, ExprT> sum(const Exp<ExprT>& expr);

    template<typename ExprT>
    LazyAllReducer<mshadow::red::minimum, ExprT> min(const Exp<ExprT>& expr);

    template<typename ExprT>
    LazyAllReducer<mshadow::red::maximum, ExprT> max(const Exp<ExprT>& expr);

    template<typename ExprT>
    LazyAxisReducer<mshadow::red::sum, ExprT, false> sum(const Exp<ExprT>& expr, const int& axis, bool keepdims=false);

    template<typename ExprT>
    LazyAxisReducer<mshadow::red::maximum, ExprT, true> argmax(const Exp<ExprT>& expr, const int& axis, bool keepdims=false);

    template<typename ExprT>
    LazyAxisReducer<mshadow::red::maximum, ExprT, false> max(const Exp<ExprT>& expr, const int& axis, bool keepdims=false);

    template<typename ExprT>
    LazyAxisReducer<mshadow::red::minimum, ExprT, true> argmin(const Exp<ExprT>& expr, const int& axis, bool keepdims=false);

    template<typename ExprT>
    LazyAxisReducer<mshadow::red::minimum, ExprT, false> min(const Exp<ExprT>& expr, const int& axis, bool keepdims=false);

}  // namespace lazy

#include "dali/array/lazy/reducers-impl.h"

#endif
