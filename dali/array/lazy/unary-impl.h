#include "dali/array/function/lazy_function.h"
#include "dali/array/functor.h"

template<template<class>class Functor, typename ExprT>
struct LazyUnary : public LazyFunction<LazyUnary<Functor,ExprT>, ExprT> {
    ExprT expr;

    LazyUnary(ExprT expr_) : LazyFunction<LazyUnary<Functor,ExprT>, ExprT>(expr_),
                             expr(expr_) {
    }

    template<int devT,typename T>
    auto to_mshadow_expr(memory::Device device, const std::vector<int>& output_shape) const ->
            decltype(
                mshadow::expr::F<Functor<T>>(
                     MshadowWrapper<devT,T,ExprT>::wrap(expr, device, output_shape)
                )
            ) {
        auto left_expr = MshadowWrapper<devT,T,ExprT>::wrap(expr, device, output_shape);
        return mshadow::expr::F<Functor<T>>(left_expr);
    }
};

namespace lazy {
    template<template<class>class Functor, typename ExprT>
    LazyUnary<Functor,ExprT> F(const Exp<ExprT>& expr) {
        return LazyUnary<Functor,ExprT>(expr.self());
    }

    template<typename ExprT>
    LazyUnary<functor::identity,ExprT> identity(const Exp<ExprT>& expr) {
        return LazyUnary<functor::identity,ExprT>(expr.self());
    }

    template<typename ExprT>
    LazyUnary<functor::sigmoid,ExprT> sigmoid(const Exp<ExprT>& expr) {
        return LazyUnary<functor::sigmoid,ExprT>(expr.self());
    }

    template<typename ExprT>
    LazyUnary<functor::tanh,ExprT> tanh(const Exp<ExprT>& expr) {
        return LazyUnary<functor::tanh,ExprT>(expr.self());
    }

    template<typename ExprT>
    LazyUnary<functor::relu,ExprT> relu(const Exp<ExprT>& expr) {
        return LazyUnary<functor::relu,ExprT>(expr.self());
    }

    template<typename ExprT>
    LazyUnary<functor::log_or_zero,ExprT> log_or_zero(const Exp<ExprT>& expr) {
        return LazyUnary<functor::log_or_zero,ExprT>(expr.self());
    }

    template<typename ExprT>
    LazyUnary<functor::abs,ExprT> abs(const Exp<ExprT>& expr) {
        return LazyUnary<functor::abs,ExprT>(expr.self());
    }

    template<typename ExprT>
    LazyUnary<functor::sign,ExprT> sign(const Exp<ExprT>& expr) {
        return LazyUnary<functor::sign,ExprT>(expr.self());
    }

    template<typename ExprT>
    LazyUnary<functor::square,ExprT> square(const Exp<ExprT>& expr) {
        return LazyUnary<functor::square,ExprT>(expr.self());
    }
}  // namespace lazy
