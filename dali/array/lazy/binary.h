#ifndef DALI_ARRAY_LAZY_BINARY_H
#define DALI_ARRAY_LAZY_BINARY_H

#include "dali/array/functor.h"

template<template<class>class Functor, typename LeftT, typename RightT>
struct LazyBinary;

template<typename LeftT, typename RightT>
struct LazyOuter;

template<template<class>class Functor, typename LeftT, typename RightT>
struct LazyBinaryIndexed;

namespace lazy {
    template<template<class>class Functor, typename T1, typename T2>
    LazyBinary<Functor, T1, T2> F(const T1& expr, const T2& expr2);

    template<typename T1, typename T2>
    LazyOuter<T1, T2> outer(const T1& expr, const T2& expr2);

    template <typename T, typename T2>
    LazyBinary<functor::add, T, T2> add(const T& a, const T2& b);

    template <typename T, typename T2>
    LazyBinary<functor::sub, T, T2> sub(const T& a, const T2& b);

    template <typename T, typename T2>
    LazyBinary<functor::eltmul, T, T2> eltmul(const T& a, const T2& b);

    template <typename T, typename T2>
    LazyBinary<functor::eltdiv, T, T2> eltdiv(const T& a, const T2& b);

    template <typename T, typename T2>
    LazyBinary<functor::lessthanequal, T, T2> lessthanequal(const T& a, const T2& b);

    template <typename T, typename T2>
    LazyBinary<functor::greaterthanequal, T, T2> greaterthanequal(const T& a, const T2& b);

    template <typename T, typename T2>
    LazyBinary<functor::power, T, T2> pow(const T& a, const T2& b);

    template <typename T, typename T2>
    LazyBinary<functor::max_scalar, T, T2> eltmax(const T& a, const T2& b);

    template <typename T, typename T2>
    LazyBinary<functor::clip, T, T2> clip(const T& a, const T2& b);

    template <typename T, typename T2>
    LazyBinary<functor::min_scalar, T, T2> eltmin(const T& a, const T2& b);

    template <typename T, typename T2>
    LazyBinary<functor::binary_cross_entropy, T, T2> binary_cross_entropy(const T& a, const T2& b);

    template <typename T, typename T2>
    LazyBinary<functor::binary_cross_entropy_grad, T, T2> binary_cross_entropy_grad(const T& a, const T2& b);

    template<typename T, typename T2>
    LazyBinary<functor::equals, T, T2> equals(const T& a, const T2& b);

    template<typename T, typename T2>
    LazyBinary<functor::prelu, T, T2> prelu(const T& x, const T2& weights);

    template<typename T, typename T2>
    LazyBinary<functor::prelu_backward_weights, T, T2> prelu_backward_weights(const T& x, const T2& grad);

    template<typename T, typename T2>
    LazyBinary<functor::prelu_backward_inputs, T, T2> prelu_backward_inputs(const T& x, const T2& weights);

}  // namespace lazy

#include "dali/array/lazy/binary-impl.h"

#endif
