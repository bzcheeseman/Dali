#include "dali/array/function/args/mshadow_wrapper.h"
#include "dali/array/function/expression.h"
#include "dali/array/function/lazy_evaluator.h"

template<class Functor, typename ExprT>
struct LazyReducer : public LazyExp<LazyReducer<Functor,ExprT>> {
    typedef LazyReducer<Functor,ExprT> self_t;

    ExprT expr;
    std::vector<int> shape_;
    DType dtype_;

    LazyReducer(const ExprT& _expr) : expr(_expr), shape_({}) {
        bool dtype_good;
        std::tie(dtype_good, dtype_) = LazyCommonPropertyExtractor<DTypeProperty>::extract_unary(expr);
        ASSERT2(dtype_good, "LazyReducer function called on dtypeless expression.");
    }

    const std::vector<int>& shape() const {
        return shape_;
    }

    const DType& dtype() const {
        return dtype_;
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

    AssignableArray as_assignable() const {
        auto res = LazyEvaluator<self_t>::run(*this);
        return res;
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
