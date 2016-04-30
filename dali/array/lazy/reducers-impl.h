#include <vector>

#include "dali/array/function/args/mshadow_wrapper.h"
#include "dali/array/function/lazy_function.h"

template<class Functor, typename ExprT>
struct LazyReducer : public LazyFunction<LazyReducer<Functor,ExprT>, ExprT> {
    ExprT expr;

    static std::vector<int> lazy_output_shape(const ExprT&) {
        return {};
    }

    LazyReducer(const ExprT& expr_) :
            LazyFunction<LazyReducer<Functor,ExprT>, ExprT>(expr_),
            expr(expr_) {
    }

    template<int devT,typename T>
    auto to_mshadow_expr(memory::Device device) const ->
            decltype(
                Functor::reduce(
                    MshadowWrapper<devT,T,decltype(expr)>::wrap(expr, device)
                )
            ) {
        auto left_expr  = MshadowWrapper<devT, T, decltype(expr)>::wrap(expr, device);
        auto ret = Functor::reduce(left_expr);
        return ret;

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
    // template<template<class>class Functor, typename ExprT>
    // LazyElementwise<Functor,ExprT> F(const Exp<ExprT>& expr) {
    //     return LazyElementwise<Functor,ExprT>(expr.self());
    // }

    template<typename ExprT>
    LazyReducer<myops::sum_all, ExprT> sum_all(const Exp<ExprT>& expr) {
        return LazyReducer<myops::sum_all, ExprT>(expr.self());
    }
}  // namespace lazy
