#ifndef DALI_ARRAY_LAZY_UNARY_H
#define DALI_ARRAY_LAZY_UNARY_H

#include "dali/array/functor.h"

template<template<class>class Functor, typename ExprT>
struct LazyUnary;

template<template<class>class Functor, typename ExprT>
struct LazyUnaryIndexed;

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
    LazyUnary<functor::exp,ExprT> exp(const Exp<ExprT>& expr);

    template<typename ExprT>
    LazyUnary<functor::inv,ExprT> eltinv(const Exp<ExprT>& expr);

    template<typename ExprT>
    LazyUnary<functor::softplus,ExprT> softplus(const Exp<ExprT>& expr);

    template<typename ExprT>
    LazyUnary<functor::relu,ExprT> relu(const Exp<ExprT>& expr);

    template<typename ExprT>
    LazyUnary<functor::log,ExprT> log(const Exp<ExprT>& expr);

    template<typename ExprT>
    LazyUnary<functor::negative_log,ExprT> negative_log(const Exp<ExprT>& expr);

    template<typename ExprT>
    LazyUnary<functor::log_or_zero,ExprT> log_or_zero(const Exp<ExprT>& expr);

    template<typename ExprT>
    LazyUnary<functor::abs,ExprT> abs(const Exp<ExprT>& expr);

    template<typename ExprT>
    LazyUnary<functor::sign,ExprT> sign(const Exp<ExprT>& expr);

    template<typename ExprT>
    LazyUnary<functor::square,ExprT> square(const Exp<ExprT>& expr);

    template<typename ExprT>
    LazyUnary<functor::cube,ExprT> cube(const Exp<ExprT>& expr);

    template<typename ExprT>
    LazyUnary<functor::sqrt_f,ExprT> sqrt(const Exp<ExprT>& expr);

    // Reciprocal square root f(x) = x ^ -0.5
    template<typename ExprT>
    LazyUnary<functor::rsqrt,ExprT> rsqrt(const Exp<ExprT>& expr);
}  // namespace lazy

#include "dali/array/lazy/unary-impl.h"

#endif
