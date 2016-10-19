#ifndef DALI_ARRAY_LAZY_BINARY_H
#define DALI_ARRAY_LAZY_BINARY_H

#include "dali/array/functor.h"

template<template<class>class Functor, typename LeftT, typename RightT>
struct LazyBinary;

namespace lazy {
    template<template<class>class Functor, typename T1, typename T2>
    LazyBinary<Functor, T1, T2> F(const T1& expr, const T2& expr2);

    template <typename T, typename T2>
    LazyBinary<functor::add, T, T2> add(const T& a, const T2& b);

    template <typename T, typename T2>
    LazyBinary<functor::sub, T, T2> sub(const T& a, const T2& b);

    template <typename T, typename T2>
    LazyBinary<functor::eltmul, T, T2> eltmul(const T& a, const T2& b);

    template <typename T, typename T2>
    LazyBinary<functor::eltdiv, T, T2> eltdiv(const T& a, const T2& b);

    template<typename T, typename T2>
    LazyBinary<functor::equals, T, T2> equals(const T& a, const T2& b);

}  // namespace lazy

#include "dali/array/lazy/binary-impl.h"

#endif
