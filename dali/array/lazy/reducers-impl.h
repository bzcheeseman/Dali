#include <vector>

#include "dali/array/shape.h"
#include "dali/array/function/lazy_function.h"

template<class Functor, typename ExprT>
struct LazyReducer : public LazyFunction<LazyReducer<Functor,ExprT>, ExprT> {
    static const int evaluation_dim;
    ExprT expr;

    static std::vector<int> lazy_output_bshape(const ExprT&) {
        return {};
    }

    LazyReducer(const ExprT& expr_) :
            LazyFunction<LazyReducer<Functor,ExprT>, ExprT>(expr_),
            expr(expr_) {
    }

    template<int devT,typename T>
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

template<class Functor, typename ExprT>
const int LazyReducer<Functor,ExprT>::evaluation_dim = 1;

namespace myops {
    struct sum_all {
        template<typename T>
        static inline auto reduce(const T& expr) -> decltype(mshadow::expr::sum_all(expr)) {
            return mshadow::expr::sum_all(expr);
        }
    };
}

namespace lazy {
    // template<template<class>class Functor, typename ExprT>
    // LazyElementwise<Functor,ExprT> F(const Exp<ExprT>& expr) {
    //     return LazyElementwise<Functor,ExprT>(expr.self());
    // }

    template<typename ExprT>
    LazyReducer<myops::sum_all, ExprT> sum_all(const Exp<ExprT>& expr) {
        return LazyReducer<myops::sum_all, ExprT>(expr.self());
    }
}  // namespace lazy
