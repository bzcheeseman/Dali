#ifndef DALI_ARRAY_LAZY_UNARY_H
#define DALI_ARRAY_LAZY_UNARY_H

#include "dali/array/TensorFunctions.h"

template<template<class>class Functor, typename ExprT>
struct LazyUnary;

namespace lazy {
    template<template<class>class Functor, typename ExprT>
    LazyUnary<Functor,ExprT> F(const Exp<ExprT>& expr);

    template<typename ExprT>
    LazyUnary<tensor_ops::op::sigmoid,ExprT> sigmoid(const Exp<ExprT>& expr);

    template<typename ExprT>
    LazyUnary<tensor_ops::op::tanh,ExprT> tanh(const Exp<ExprT>& expr);

    template<typename ExprT>
    LazyUnary<tensor_ops::op::relu,ExprT> relu(const Exp<ExprT>& expr);

    template<typename ExprT>
    LazyUnary<tensor_ops::op::log_or_zero,ExprT> log_or_zero(const Exp<ExprT>& expr);

    template<typename ExprT>
    LazyUnary<tensor_ops::op::abs,ExprT> abs(const Exp<ExprT>& expr);

    template<typename ExprT>
    LazyUnary<tensor_ops::op::sign,ExprT> sign(const Exp<ExprT>& expr);
}  // namespace lazy

#include "dali/array/lazy/unary-impl.h"

#endif
