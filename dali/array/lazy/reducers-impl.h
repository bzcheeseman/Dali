#include <vector>

#include "dali/array/shape.h"
#include "dali/array/function/lazy_function.h"
#include "dali/array/lazy/base_lazy_axis_reducer.h"

#include <mshadow/extension/reduce_with_axis.h>

template<class Functor, typename ExprT>
struct LazyAllReducer : public LazyFunctionNonRecusive<LazyAllReducer<Functor,ExprT>, ExprT> {
    static const int evaluation_dim;
    ExprT expr;

    static std::vector<int> lazy_output_bshape(const ExprT&) {
        return {};
    }

    LazyAllReducer(const ExprT& expr_) :
            LazyFunctionNonRecusive<LazyAllReducer<Functor,ExprT>, ExprT>(expr_),
            expr(expr_) {
    }

    template<int devT, typename T>
    auto to_mshadow_expr(memory::Device device, const std::vector<int>& output_shape) const ->
            decltype(
                Functor::reduce(
                    MshadowWrapper<devT,T,decltype(expr)>::wrap(expr, device, output_shape)
                )
            ) {

        auto left_expr  =
                MshadowWrapper<devT, T, decltype(expr)>::wrap(
                        expr, device, bshape2shape(expr.bshape())
                );
        auto ret = Functor::reduce(left_expr);
        return ret;
    }
};

template<class Functor, typename ExprT, bool return_indices>
struct LazyAxisReducer {
};

template<class Functor, typename ExprT>
const int LazyAllReducer<Functor, ExprT>::evaluation_dim = 1;


template<class Functor, typename ExprT>
struct LazyAxisReducer<Functor, ExprT, false> : public BaseLazyAxisReducer<LazyFunction, LazyAxisReducer<Functor, ExprT, false>, ExprT, Functor, false> {
    using BaseLazyAxisReducer<LazyFunction, LazyAxisReducer<Functor, ExprT, false>, ExprT, Functor, false>::BaseLazyAxisReducer; // inherit parent constructor
};

template<class Functor, typename ExprT>
struct LazyAxisReducer<Functor, ExprT, true> : public BaseLazyAxisReducer<LazyFunction, LazyAxisReducer<Functor, ExprT, true>, ExprT, Functor, true> {
    using BaseLazyAxisReducer<LazyFunction, LazyAxisReducer<Functor, ExprT, true>, ExprT, Functor, true>::BaseLazyAxisReducer; // inherit parent constructor

    static DType lazy_output_dtype(const ExprT& expr_, const int& reduce_axis_, bool) {
        return DTYPE_INT32;
    }
};

namespace myops {
    struct sum_all {
        template<typename T>
        static inline auto reduce(const T& expr) -> decltype(mshadow::expr::sum_all(expr)) {
            return mshadow::expr::sum_all(expr);
        }
    };
}

namespace lazy {
    template<typename ExprT>
    LazyAllReducer<myops::sum_all, ExprT> sum_all(const Exp<ExprT>& expr) {
        return LazyAllReducer<myops::sum_all, ExprT>(expr.self());
    }

    template<typename ExprT>
    LazyAxisReducer<mshadow::red::sum, ExprT, false> sum_axis(const Exp<ExprT>& expr, const int& axis, bool keepdims) {
        return LazyAxisReducer<mshadow::red::sum, ExprT, false>(expr.self(), axis, keepdims);
    }

    template<typename ExprT>
    LazyAxisReducer<mshadow::red::maximum, ExprT, true> argmax_axis(const Exp<ExprT>& expr, const int& axis, bool keepdims) {
        return LazyAxisReducer<mshadow::red::maximum, ExprT, true>(expr.self(), axis, keepdims);
    }

    template<typename ExprT>
    LazyAxisReducer<mshadow::red::maximum, ExprT, false> max_axis(const Exp<ExprT>& expr, const int& axis, bool keepdims) {
        return LazyAxisReducer<mshadow::red::maximum, ExprT, false>(expr.self(), axis, keepdims);
    }

    template<typename ExprT>
    LazyAxisReducer<mshadow::red::minimum, ExprT, true> argmin_axis(const Exp<ExprT>& expr, const int& axis, bool keepdims) {
        return LazyAxisReducer<mshadow::red::minimum, ExprT, true>(expr.self(), axis, keepdims);
    }

    template<typename ExprT>
    LazyAxisReducer<mshadow::red::minimum, ExprT, false> min_axis(const Exp<ExprT>& expr, const int& axis, bool keepdims) {
        return LazyAxisReducer<mshadow::red::minimum, ExprT, false>(expr.self(), axis, keepdims);
    }
}  // namespace lazy
