#ifndef DALI_ARRAY_LAZY_UNARY_H
#define DALI_ARRAY_LAZY_UNARY_H

#include "dali/array/functor.h"

template<template<class>class Functor, typename ExprT>
struct LazyUnary;

namespace lazy {
    template<typename ExprT>
    LazyUnary<functor::identity,ExprT> identity(const Exp<ExprT>& expr);
}  // namespace lazy

#include "dali/array/lazy/unary-impl.h"

#endif
