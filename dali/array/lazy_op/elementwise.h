#ifndef DALI_ARRAY_LAZY_OP_ELEMENTWISE_H
#define DALI_ARRAY_LAZY_OP_ELEMENTWISE_H

template<template<class>class Functor, typename ExprT>
struct LazyElementwise : public RValueExp<LazyElementwise<Functor,ExprT>> {
    typedef LazyElementwise<Functor,ExprT> self_t;

    ExprT expr;
    std::vector<int> shape_;
    DType dtype_;

    LazyElementwise(const ExprT& _expr) :
            expr(_expr) {
        bool shape_good, dtype_good;
        std::tie(shape_good, shape_) = LazyCommonPropertyExtractor<ShapeProperty>::extract_unary(expr);
        std::tie(dtype_good, dtype_) = LazyCommonPropertyExtractor<DTypeProperty>::extract_unary(expr);
        ASSERT2(shape_good, "Elementwise function called on shapeless expression.");
        ASSERT2(dtype_good, "Elementwise function called on dtypeless expression.");
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
                mshadow::expr::F<Functor<T>>(
                     MshadowWrapper<devT,T,decltype(expr)>::to_expr(expr, device)
                )
            ) {
        auto left_expr  = MshadowWrapper<devT,T,decltype(expr)>::to_expr(expr,  device);
        return mshadow::expr::F<Functor<T>>(left_expr);
    }

    AssignableArray as_assignable() const {
        return Evaluator<self_t>::run(*this);
    }
};


namespace lazy {
    template<template<class>class Functor, typename ExprT>
    LazyElementwise<Functor,ExprT> F(const Exp<ExprT>& expr) {
        return LazyElementwise<Functor,ExprT>(expr.self());
    }

    template<typename ExprT>
    LazyElementwise<TensorOps::op::sigmoid,ExprT> sigmoid(const Exp<ExprT>& expr) {
        return LazyElementwise<TensorOps::op::sigmoid,ExprT>(expr.self());
    }

    template<typename ExprT>
    LazyElementwise<TensorOps::op::tanh,ExprT> tanh(const Exp<ExprT>& expr) {
        return LazyElementwise<TensorOps::op::tanh,ExprT>(expr.self());
    }

    template<typename ExprT>
    LazyElementwise<TensorOps::op::relu,ExprT> relu(const Exp<ExprT>& expr) {
        return LazyElementwise<TensorOps::op::relu,ExprT>(expr.self());
    }

    template<typename ExprT>
    LazyElementwise<TensorOps::op::log_or_zero,ExprT> log_or_zero(const Exp<ExprT>& expr) {
        return LazyElementwise<TensorOps::op::log_or_zero,ExprT>(expr.self());
    }

    template<typename ExprT>
    LazyElementwise<TensorOps::op::abs,ExprT> abs(const Exp<ExprT>& expr) {
        return LazyElementwise<TensorOps::op::abs,ExprT>(expr.self());
    }

    template<typename ExprT>
    LazyElementwise<TensorOps::op::sign,ExprT> sign(const Exp<ExprT>& expr) {
        return LazyElementwise<TensorOps::op::sign,ExprT>(expr.self());
    }
}
#endif
