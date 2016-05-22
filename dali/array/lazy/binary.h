#ifndef DALI_ARRAY_LAZY_BINARY_H
#define DALI_ARRAY_LAZY_BINARY_H

#include "dali/array/functor.h"

template<template<class>class Functor, typename LeftT, typename RightT>
struct LazyBinary;

namespace lazy {
    template<template<class>class Functor, typename T, typename T2>
    LazyBinary<Functor, T, T2> F(const Exp<T>& expr, const T2& expr2);

    template <typename T, typename T2>
    LazyBinary<functor::add, T, T2> add(T a, T2 b);

    template <typename T, typename T2>
    LazyBinary<functor::sub, T, T2> sub(T a, T2 b);

    template <typename T, typename T2>
    LazyBinary<functor::eltmul, T, T2> eltmul(T a, T2 b);

    template <typename T, typename T2>
    LazyBinary<functor::eltdiv, T, T2> eltdiv(T a, T2 b);

    template <typename T, typename T2>
    LazyBinary<functor::power, T, T2> pow(T a, T2 b);

    template <typename T, typename T2>
    LazyBinary<functor::max_scalar, T, T2> eltmax(T a, T2 b);

    template <typename T, typename T2>
    LazyBinary<functor::clip, T, T2> clip(T a, T2 b);

    template <typename T, typename T2>
    LazyBinary<functor::min_scalar, T, T2> eltmin(T a, T2 b);

    template <typename T, typename T2>
    LazyBinary<functor::binary_cross_entropy, T, T2> binary_cross_entropy(T a, T2 b);

    template <typename T, typename T2>
    LazyBinary<functor::binary_cross_entropy_grad, T, T2> binary_cross_entropy_grad(T a, T2 b);

    template<typename T, typename T2>
    LazyBinary<functor::subsample_partial_grad, T, T2> subsample_partial_grad(T a, T2 b);


}  // namespace lazy

#include "dali/array/lazy/binary-impl.h"

#endif
