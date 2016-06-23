#ifndef DALI_ARRAY_FUNCTION_ARGS_REDUCE_OVER_LAZY_EXPR_H
#define DALI_ARRAY_FUNCTION_ARGS_REDUCE_OVER_LAZY_EXPR_H

#include <tuple>
#include "dali/array/function/expression.h"

template<int data_format, typename SrcExp>
struct LazyIm2Col;

template<template<class>class Functor, typename LeftT, typename RightT>
struct LazyBinary;

template<typename LeftT, typename RightT>
struct LazyOuter;

template<template<class>class Functor, typename LeftT, typename RightT>
struct LazyBinaryIndexed;

template<template<class>class Functor, typename ExprT>
struct LazyUnary;

template<template<class>class Functor, typename ExprT>
struct LazyUnaryIndexed;

template<class Functor, typename ExprT>
struct LazyAllReducer;

template<typename ExprT, typename NewT>
struct LazyCast;

template<class Functor, typename ExprT, bool return_indices>
struct LazyAxisReducer;

template<typename ExprT1, typename ExprT2>
struct LazyTake;

template<typename ExprT1, typename ExprT2>
struct LazyTakeFromRows;

namespace internal {
    template<typename ExprT>
    struct NonRecursiveLazySumAxis;
}  // namespace internal

template<typename Reducer>
struct ReduceOverLazyExpr {
    typedef std::tuple<typename Reducer::outtype_t, typename Reducer::state_t> outtuple_t;

    template<template<class>class Functor, typename LeftT, typename RightT, typename... Args>
    static outtuple_t unfold_helper(
            const outtuple_t& state,
            const LazyBinary<Functor, LeftT,RightT>& binary_expr,
            const Args&... args) {
        return unfold_helper(state, binary_expr.left, binary_expr.right, args...);
    }

    template<typename LeftT, typename RightT, typename... Args>
    static outtuple_t unfold_helper(
            const outtuple_t& state,
            const LazyOuter<LeftT,RightT>& binary_expr,
            const Args&... args) {
        return unfold_helper(state, binary_expr.left, binary_expr.right, args...);
    }

    template<template<class>class Functor, typename LeftT, typename RightT, typename... Args>
    static outtuple_t unfold_helper(
            const outtuple_t& state,
            const LazyBinaryIndexed<Functor, LeftT,RightT>& binary_expr,
            const Args&... args) {
        return unfold_helper(state, binary_expr.left, binary_expr.right, args...);
    }

    template<typename ExprT, typename... Args>
    static outtuple_t unfold_helper(
            const outtuple_t& state,
            const internal::NonRecursiveLazySumAxis<ExprT>& reducer_expr,
            const Args&... args) {
        return unfold_helper(state, reducer_expr.expr, args...);
    }

    template<int data_format, typename ExprT, typename... Args>
    static outtuple_t unfold_helper(
            const outtuple_t& state,
            const LazyIm2Col<data_format,ExprT>& im2col_expr,
            const Args&... args) {
        return unfold_helper(state, im2col_expr.src, args...);
    }

    template<typename ExprT, typename NewType, typename... Args>
    static outtuple_t unfold_helper(
            const outtuple_t& state,
            const LazyCast<ExprT, NewType>& reducer_expr,
            const Args&... args) {
        return unfold_helper(state, reducer_expr.expr, args...);
    }

    template<template<class>class Functor, typename ExprT, typename... Args>
    static outtuple_t unfold_helper(
            const outtuple_t& state,
            const LazyUnary<Functor,ExprT>& elementwise_expr,
            const Args&... args) {
        return unfold_helper(state, elementwise_expr.expr, args...);
    }

    template<template<class>class Functor, typename ExprT, typename... Args>
    static outtuple_t unfold_helper(
            const outtuple_t& state,
            const LazyUnaryIndexed<Functor,ExprT>& elementwise_expr,
            const Args&... args) {
        return unfold_helper(state, elementwise_expr.expr, args...);
    }

    template<typename ExprT1, typename ExprT2, typename... Args>
    static outtuple_t unfold_helper(
            const outtuple_t& state,
            const LazyTake<ExprT1, ExprT2>& take_expr,
            const Args&... args) {
        return unfold_helper(state, take_expr.src, take_expr.indices, args...);
    }

    template<typename ExprT1, typename ExprT2, typename... Args>
    static outtuple_t unfold_helper(
            const outtuple_t& state,
            const LazyTakeFromRows<ExprT1, ExprT2>& take_expr,
            const Args&... args) {
        return unfold_helper(state, take_expr.src, take_expr.indices, args...);
    }

    template<class Functor, typename ExprT, typename... Args>
    static outtuple_t unfold_helper(
            const outtuple_t& state,
            const LazyAllReducer<Functor,ExprT>& reducer_expr,
            const Args&... args) {
        return unfold_helper(state, reducer_expr.expr, args...);
    }

    template<class Functor, typename ExprT, bool return_indices, typename... Args>
    static outtuple_t unfold_helper(
            const outtuple_t& state,
            const LazyAxisReducer<Functor,ExprT,return_indices>& reducer_expr,
            const Args&... args) {
        return unfold_helper(state, reducer_expr.expr, args...);
    }

    template<typename T, typename... Args>
    static outtuple_t unfold_helper(const outtuple_t& state, const T& arg, const Args&... args) {
        static_assert(!std::is_base_of<LazyExpType,T>::value,
                "Every lazy expression needs to have a corresponding `unfold_helper` method in " __FILE__
                ". Did you add a new lazy expression and forget to implement `unfold_helper`?");
        return unfold_helper(Reducer::reduce_step(state, arg), args...);
    }

    static outtuple_t unfold_helper(const outtuple_t& state) {
        return state;
    }

    template<typename... Args>
    static typename Reducer::outtype_t reduce(const Args&... args) {
        auto initial_tuple = outtuple_t();
        return std::get<0>(unfold_helper(initial_tuple, args...));
    }
};

#endif // DALI_ARRAY_FUNCTION_ARGS_REDUCE_OVER_LAZY_EXPR_H
