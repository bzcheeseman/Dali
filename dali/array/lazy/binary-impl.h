#include "dali/array/function/args/mshadow_wrapper.h"
#include "dali/array/function/args/property_reducer.h"
#include "dali/array/function/expression.h"
#include "dali/array/function/lazy_evaluator.h"
#include "dali/array/TensorFunctions.h"

template<template<class>class Functor, typename LeftT, typename RightT>
struct LazyBinary : public LazyExp<LazyBinary<Functor,LeftT,RightT>> {
    typedef LazyBinary<Functor,LeftT,RightT> self_t;
    LeftT  left;
    RightT right;
    std::vector<int> shape_;
    DType dtype_;

    LazyBinary(const LeftT& _left, const RightT& _right) :
            left(_left),
            right(_right),
            shape_(std::get<1>(LazyCommonPropertyExtractor<ShapeProperty>::extract_binary(left, right))),
            dtype_(std::get<1>(LazyCommonPropertyExtractor<DTypeProperty>::extract_binary(left, right))) {
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
                     MshadowWrapper<devT,T,decltype(left)>::wrap(left, device),
                     MshadowWrapper<devT,T,decltype(right)>::wrap(right, device)
                )
            ) {
        auto left_expr  = MshadowWrapper<devT,T,decltype(left)>::wrap(left,  device);
        auto right_expr = MshadowWrapper<devT,T,decltype(right)>::wrap(right, device);
        return mshadow::expr::F<Functor<T>>(left_expr, right_expr);
    }

    AssignableArray as_assignable() const {
        return LazyEvaluator<self_t>::run(*this);
    }
};


namespace lazy {
    template <typename T, typename T2>
    LazyBinary<tensor_ops::op::add, T, T2> add(T a, T2 b) {
        return LazyBinary<tensor_ops::op::add, T, T2>(a, b);
    }

    template <typename T, typename T2>
    LazyBinary<tensor_ops::op::sub, T, T2> sub(T a, T2 b) {
        return LazyBinary<tensor_ops::op::sub, T, T2>(a, b);
    }

    template <typename T, typename T2>
    LazyBinary<tensor_ops::op::eltmul, T, T2> eltmul(T a, T2 b) {
        return LazyBinary<tensor_ops::op::eltmul, T, T2>(a, b);
    }

    template <typename T, typename T2>
    LazyBinary<tensor_ops::op::eltdiv, T, T2> eltdiv(T a, T2 b) {
        return LazyBinary<tensor_ops::op::eltdiv, T, T2>(a, b);
    }
}  // namespace lazy
