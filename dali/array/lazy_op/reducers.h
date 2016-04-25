#ifndef DALI_ARRAY_LAZY_OP_REDUCERS_H
#define DALI_ARRAY_LAZY_OP_REDUCERS_H

template<typename OP, typename TA, typename DType, int ta>
inline auto
ReducerF(const mshadow::expr::Exp<TA, DType, ta> &src) -> decltype(OP::Reduce(src)) {
    return OP::Reduce(src);
}

template<class Functor, typename ExprT>
struct LazyReducer : public RValueExp<LazyReducer<Functor,ExprT>> {
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
                ReducerF<Functor>(
                    MshadowWrapper<devT,T,decltype(expr)>::to_expr(expr, device)
                )
            ) {
        auto left_expr  = MshadowWrapper<devT, T, decltype(expr)>::to_expr(expr, device);
        return ReducerF<Functor>(left_expr);
    }

    AssignableArray as_assignable() const {
        return Evaluator<self_t>::run(*this);
    }
};

namespace myops {
    struct sum_all {
        template<typename T>
        static inline auto Reduce(const T& expr) -> decltype(mshadow::expr::sum_all(expr)) {
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
}

#endif
