#include "dali/array/function/lazy_function.h"
#include "dali/array/functor.h"

template<template<class>class Functor, typename LeftT, typename RightT>
struct LazyBinary : public LazyFunction<LazyBinary<Functor,LeftT,RightT>, LeftT, RightT> {
    LeftT  left;
    RightT right;

    LazyBinary(const LeftT& left_, const RightT& right_) :
            LazyFunction<LazyBinary<Functor,LeftT,RightT>, LeftT, RightT>(left_, right_),
            left(left_),
            right(right_) {
    }

    template<int devT,typename T>
    auto to_mshadow_expr(memory::Device device, const std::vector<int>& output_shape) const ->
            decltype(
                mshadow::expr::F<Functor<T>>(
                     MshadowWrapper<devT,T,decltype(left)>::wrap(left, device, output_shape),
                     MshadowWrapper<devT,T,decltype(right)>::wrap(right, device, output_shape)
                )
            ) {
        auto left_expr  = MshadowWrapper<devT,T,decltype(left)>::wrap(left,  device, output_shape);
        auto right_expr = MshadowWrapper<devT,T,decltype(right)>::wrap(right, device, output_shape);
        return mshadow::expr::F<Functor<T>>(left_expr, right_expr);
    }
};

namespace lazy {
    #define DALI_LAZY_IMPLEMENT_LAZYBINARY_EXPR(FUNCNAME, KERNELNAME)\
        template <typename T, typename T2>\
        LazyBinary<KERNELNAME, T, T2> FUNCNAME(T a, T2 b) {\
            return LazyBinary<KERNELNAME, T, T2>(a, b);\
        }\

    DALI_LAZY_IMPLEMENT_LAZYBINARY_EXPR(add, functor::add);
    DALI_LAZY_IMPLEMENT_LAZYBINARY_EXPR(sub, functor::sub);
    DALI_LAZY_IMPLEMENT_LAZYBINARY_EXPR(eltmul, functor::eltmul);
    DALI_LAZY_IMPLEMENT_LAZYBINARY_EXPR(eltdiv, functor::eltdiv);
    DALI_LAZY_IMPLEMENT_LAZYBINARY_EXPR(pow, functor::power);
    DALI_LAZY_IMPLEMENT_LAZYBINARY_EXPR(binary_cross_entropy, functor::binary_cross_entropy);
    DALI_LAZY_IMPLEMENT_LAZYBINARY_EXPR(binary_cross_entropy_grad, functor::binary_cross_entropy_grad);
    DALI_LAZY_IMPLEMENT_LAZYBINARY_EXPR(eltmax, functor::max_scalar);
    DALI_LAZY_IMPLEMENT_LAZYBINARY_EXPR(eltmin, functor::min_scalar);
    DALI_LAZY_IMPLEMENT_LAZYBINARY_EXPR(subsample_partial_grad, functor::subsample_partial_grad);
}  // namespace lazy
