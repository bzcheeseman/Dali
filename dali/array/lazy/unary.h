#ifndef DALI_ARRAY_LAZY_UNARY_H
#define DALI_ARRAY_LAZY_UNARY_H

#include "dali/array/functor.h"

template<template<class>class Functor, typename ExprT>
struct LazyUnary;

namespace lazy {
    template<template<class>class Functor, typename ExprT>
    LazyUnary<Functor,ExprT> F(const Exp<ExprT>& expr);

    template<typename ExprT>
    LazyUnary<functor::identity,ExprT> identity(const Exp<ExprT>& expr);

    template<typename ExprT>
    LazyUnary<functor::sigmoid,ExprT> sigmoid(const Exp<ExprT>& expr);

    template<typename ExprT>
    LazyUnary<functor::tanh,ExprT> tanh(const Exp<ExprT>& expr);

    template<typename ExprT>
    LazyUnary<functor::relu,ExprT> relu(const Exp<ExprT>& expr);

    template<typename ExprT>
    LazyUnary<functor::log_or_zero,ExprT> log_or_zero(const Exp<ExprT>& expr);

    template<typename ExprT>
    LazyUnary<functor::abs,ExprT> abs(const Exp<ExprT>& expr);

    template<typename ExprT>
    LazyUnary<functor::sign,ExprT> sign(const Exp<ExprT>& expr);

    template<typename ExprT>
    LazyUnary<functor::square,ExprT> square(const Exp<ExprT>& expr);
}  // namespace lazy

#include "dali/array/lazy/unary-impl.h"

#endif
