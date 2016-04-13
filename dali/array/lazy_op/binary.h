#ifndef DALI_ARRAY_LAZY_OP_BINARY_H
#define DALI_ARRAY_LAZY_OP_BINARY_H

#include "dali/array/function/property_extractor.h"
#include "dali/array/lazy_op/expression.h"
#include "dali/array/lazy_op/evaluator.h"
#include "dali/array/TensorFunctions.h"

template<template<class>class Functor, typename LeftT, typename RightT>
struct Binary : public RValueExp<Binary<Functor,LeftT,RightT>> {
    typedef Binary<Functor,LeftT,RightT> self_t;
    LeftT  left;
    RightT right;
    std::vector<int> shape_;
    DType dtype_;

    Binary(const LeftT& _left, const RightT& _right) :
            left(_left),
            right(_right),
            shape_(LazyCommonPropertyExtractor<ShapeProperty>::extract_binary(left, right)),
            dtype_(LazyCommonPropertyExtractor<DTypeProperty>::extract_binary(left, right)) {
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
                     MshadowWrapper<devT,T,decltype(left)>::to_expr(left, device),
                     MshadowWrapper<devT,T,decltype(right)>::to_expr(right, device)
                )
            ) {
        auto left_expr  = MshadowWrapper<devT,T,decltype(left)>::to_expr(left,  device);
        auto right_expr = MshadowWrapper<devT,T,decltype(right)>::to_expr(right, device);
        return mshadow::expr::F<Functor<T>>(left_expr, right_expr);
    }

    AssignableArray as_assignable() const {
        return Evaluator<self_t>::run(*this);
    }
};


namespace lazy {
    template <typename T, typename T2>
    Binary<TensorOps::op::add, T, T2> add(T a, T2 b) {
        return Binary<TensorOps::op::add, T, T2>(a, b);
    }

    template <typename T, typename T2>
    Binary<TensorOps::op::sub, T, T2> sub(T a, T2 b) {
        return Binary<TensorOps::op::sub, T, T2>(a, b);
    }

    template <typename T, typename T2>
    Binary<TensorOps::op::eltmul, T, T2> eltmul(T a, T2 b) {
        return Binary<TensorOps::op::eltmul, T, T2>(a, b);
    }

    template <typename T, typename T2>
    Binary<TensorOps::op::eltdiv, T, T2> eltdiv(T a, T2 b) {
        return Binary<TensorOps::op::eltdiv, T, T2>(a, b);
    }
}

#endif
