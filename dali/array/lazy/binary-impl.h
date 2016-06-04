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

    template<int devT,typename T, typename WrappedArrayT>
    auto to_mshadow_expr(memory::Device device, const std::vector<int>& output_shape, ArrayTransformerT<WrappedArrayT> wrap_array) const ->
            decltype(
                mshadow::expr::F<Functor<T>>(
                     MshadowWrapper<devT,T,decltype(left)>::wrap(left, device, output_shape, wrap_array),
                     MshadowWrapper<devT,T,decltype(right)>::wrap(right, device, output_shape, wrap_array)
                )
            ) {
        auto left_expr  = MshadowWrapper<devT,T,decltype(left)>::wrap(left,  device, output_shape, wrap_array);
        auto right_expr = MshadowWrapper<devT,T,decltype(right)>::wrap(right, device, output_shape, wrap_array);
        return mshadow::expr::F<Functor<T>>(left_expr, right_expr);
    }
};

template<template<class>class Functor, typename LeftT, typename RightT>
struct LazyBinaryIndexed : public LazyFunction<LazyBinaryIndexed<Functor,LeftT,RightT>, LeftT, RightT> {
    LeftT  left;
    RightT right;

    LazyBinaryIndexed(const LeftT& left_, const RightT& right_) :
            LazyFunction<LazyBinaryIndexed<Functor,LeftT,RightT>, LeftT, RightT>(left_, right_),
            left(left_),
            right(right_) {
    }

    template<int devT,typename T>
    auto to_mshadow_expr(memory::Device device, const std::vector<int>& output_shape) const ->
            decltype(
                mshadow::expr::FIndexed<Functor<T>>(
                    MshadowWrapper<devT,T,decltype(left)>::wrap(left, device, output_shape),
                    MshadowWrapper<devT,T,decltype(right)>::wrap(right, device, output_shape)
                )
            ) {
        auto left_expr  = MshadowWrapper<devT,T,decltype(left)>::wrap(left,  device, output_shape);
        auto right_expr = MshadowWrapper<devT,T,decltype(right)>::wrap(right, device, output_shape);
        return mshadow::expr::FIndexed<Functor<T>>(left_expr, right_expr);
    }
};

template<typename LeftT, typename RightT>
struct LazyOuter : public LazyFunction<LazyOuter<LeftT,RightT>, LeftT, RightT> {
    LeftT  left;
    RightT right;

    LazyOuter(const LeftT& left_, const RightT& right_) :
            LazyFunction<LazyOuter<LeftT,RightT>, LeftT, RightT>(left_, right_),
            left(left_),
            right(right_) {
    }

    static std::vector<int> lazy_output_bshape(const LeftT& left_, const RightT& right_) {
        auto left_bshape = left_.bshape();
        auto right_bshape = right_.bshape();
        ASSERT2(
            left_bshape.size() == 1 && right_bshape.size() == 1,
            utils::MS() << "inputs to outer product must be two expressions with dimensionality 1 (got left.ndim()="
                        << left_bshape.size() << ", right.ndim()=" << right_bshape.size()
                        << ").");
        return {left_bshape[0], right_bshape[0]};
    }

    template<int devT,typename T, typename WrappedArrayT>
    auto to_mshadow_expr(memory::Device device, const std::vector<int>& output_shape, ArrayTransformerT<WrappedArrayT> wrap_array) const ->
            decltype(
                mshadow::expr::outer_product(
                     MshadowWrapper<devT,T,decltype(left)>::wrap(left, device, output_shape, make_transform_array<devT,T,(int)1>()),
                     MshadowWrapper<devT,T,decltype(right)>::wrap(right, device, output_shape, make_transform_array<devT,T,(int)1>())
                )
            ) {
        auto left_expr  = MshadowWrapper<devT,T,decltype(left)>::wrap(left,   device, left.bshape(), make_transform_array<devT,T,(int)1>());
        auto right_expr = MshadowWrapper<devT,T,decltype(right)>::wrap(right, device, right.bshape(), make_transform_array<devT,T,(int)1>());
        return mshadow::expr::outer_product(left_expr, right_expr);
    }
};

namespace lazy {
    template<template<class>class Functor, typename T1, typename T2>
    LazyBinary<Functor,T1, T2> F(const T1& expr, const T2& expr2) {
        return LazyBinary<Functor, T1, T2>(expr, expr2);
    }

    #define DALI_LAZY_IMPLEMENT_LAZYBINARY_EXPR(FUNCNAME, KERNELNAME)\
        template <typename T1, typename T2>\
        LazyBinary<KERNELNAME, T1, T2> FUNCNAME(const T1& a, const T2& b) {\
            return LazyBinary<KERNELNAME, T1, T2>(a, b);\
        }\

    template<typename T1, typename T2>
    LazyOuter<T1, T2> outer(const T1& expr, const T2& expr2) {
        return LazyOuter<T1, T2>(expr, expr2);
    }

    DALI_LAZY_IMPLEMENT_LAZYBINARY_EXPR(add, functor::add);
    DALI_LAZY_IMPLEMENT_LAZYBINARY_EXPR(lessthanequal, functor::lessthanequal);
    DALI_LAZY_IMPLEMENT_LAZYBINARY_EXPR(greaterthanequal, functor::greaterthanequal);
    DALI_LAZY_IMPLEMENT_LAZYBINARY_EXPR(sub, functor::sub);
    DALI_LAZY_IMPLEMENT_LAZYBINARY_EXPR(eltmul, functor::eltmul);
    DALI_LAZY_IMPLEMENT_LAZYBINARY_EXPR(eltdiv, functor::eltdiv);
    DALI_LAZY_IMPLEMENT_LAZYBINARY_EXPR(pow, functor::power);
    DALI_LAZY_IMPLEMENT_LAZYBINARY_EXPR(binary_cross_entropy, functor::binary_cross_entropy);
    DALI_LAZY_IMPLEMENT_LAZYBINARY_EXPR(binary_cross_entropy_grad, functor::binary_cross_entropy_grad);
    DALI_LAZY_IMPLEMENT_LAZYBINARY_EXPR(eltmax, functor::max_scalar);
    DALI_LAZY_IMPLEMENT_LAZYBINARY_EXPR(eltmin, functor::min_scalar);
    DALI_LAZY_IMPLEMENT_LAZYBINARY_EXPR(clip, functor::clip);
    DALI_LAZY_IMPLEMENT_LAZYBINARY_EXPR(equals, functor::equals);
}  // namespace lazy
