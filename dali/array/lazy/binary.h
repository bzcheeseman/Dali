#ifndef DALI_ARRAY_LAZY_BINARY_H
#define DALI_ARRAY_LAZY_BINARY_H

#include "dali/array/TensorFunctions.h"

template<template<class>class Functor, typename LeftT, typename RightT>
struct LazyBinary;

namespace lazy {
    template <typename T, typename T2>
    LazyBinary<tensor_ops::op::add, T, T2> add(T a, T2 b);

    template <typename T, typename T2>
    LazyBinary<tensor_ops::op::sub, T, T2> sub(T a, T2 b);

    template <typename T, typename T2>
    LazyBinary<tensor_ops::op::eltmul, T, T2> eltmul(T a, T2 b);

    template <typename T, typename T2>
    LazyBinary<tensor_ops::op::eltdiv, T, T2> eltdiv(T a, T2 b);
}  // namespace lazy

#include "dali/array/lazy/binary-impl.h"

#endif
