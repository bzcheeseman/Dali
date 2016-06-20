#include "dali/array/function/evaluation_dim.h"
#include "dali/array/function/lazy_function.h"
#include "dali/array/functor.h"

namespace lazy {
    template<typename LeftT, typename RightT>
    struct LazyBinaryEvaluationDim {
        static const int left_value = LazyEvaluationDim<LeftT>::value;
        static const int right_value = LazyEvaluationDim<LeftT>::value;
        static const bool values_disagree = left_value != right_value;
        static const int value = (
            (left_value == -1) ?
                right_value :
                ((right_value == -1) ?
                     left_value :
                     (values_disagree ?
                        EVALUATION_DIM_ERROR :
                        left_value
                     )
                )
        );
    };
}

template<template<class>class Functor, typename LeftT, typename RightT>
struct LazyBinary : public LazyFunction<LazyBinary<Functor,LeftT,RightT>, LeftT, RightT> {
    static const int evaluation_dim;
    LeftT  left;
    RightT right;

    LazyBinary(const LeftT& left_, const RightT& right_) :
            LazyFunction<LazyBinary<Functor,LeftT,RightT>, LeftT, RightT>(left_, right_),
            left(left_),
            right(right_) {
    }

    template<int devT,typename T, int ndim>
    auto to_mshadow_expr(memory::Device device, const std::vector<int>& output_shape, const lazy::EvaluationSpec<devT, T, ndim>& wrap_array) const ->
            decltype(
                mshadow::expr::F<Functor<typename functor_helper::BinaryExtractDType<
                    decltype(MshadowWrapper<devT,T,decltype(left)>::wrap(left, device, output_shape, wrap_array)),
                    decltype(MshadowWrapper<devT,T,decltype(right)>::wrap(right, device, output_shape, wrap_array))
                >::value>
            >(
                     MshadowWrapper<devT,T,decltype(left)>::wrap(left, device, output_shape, wrap_array),
                     MshadowWrapper<devT,T,decltype(right)>::wrap(right, device, output_shape, wrap_array)
            )) {
        auto left_expr  = MshadowWrapper<devT,T,decltype(left)>::wrap(left,  device, output_shape, wrap_array);
        auto right_expr = MshadowWrapper<devT,T,decltype(right)>::wrap(right, device, output_shape, wrap_array);
        typedef typename functor_helper::BinaryExtractDType<decltype(left_expr), decltype(right_expr)>::value functor_dtype_t;
        return mshadow::expr::F<Functor<functor_dtype_t>>(left_expr, right_expr);
    }
};


template<template<class>class Functor, typename LeftT, typename RightT>
const int LazyBinary<Functor, LeftT, RightT>::evaluation_dim = lazy::LazyBinaryEvaluationDim<LeftT, RightT>::value;

template<template<class>class Functor, typename LeftT, typename RightT>
struct LazyBinaryIndexed : public LazyFunction<LazyBinaryIndexed<Functor,LeftT,RightT>, LeftT, RightT> {
    static const int evaluation_dim;
    LeftT  left;
    RightT right;

    LazyBinaryIndexed(const LeftT& left_, const RightT& right_) :
            LazyFunction<LazyBinaryIndexed<Functor,LeftT,RightT>, LeftT, RightT>(left_, right_),
            left(left_),
            right(right_) {
    }

    template<int devT,typename T, int ndim>
    auto to_mshadow_expr(memory::Device device, const std::vector<int>& output_shape, const lazy::EvaluationSpec<devT, T, ndim>& wrap_array) const ->
            decltype(
                mshadow::expr::F<mshadow::expr::FIndexed<typename functor_helper::BinaryExtractDType<
                    decltype(MshadowWrapper<devT,T,decltype(left)>::wrap(left, device, output_shape, wrap_array)),
                    decltype(MshadowWrapper<devT,T,decltype(right)>::wrap(right, device, output_shape, wrap_array))
                >::value>
            >(
                     MshadowWrapper<devT,T,decltype(left)>::wrap(left, device, output_shape, wrap_array),
                     MshadowWrapper<devT,T,decltype(right)>::wrap(right, device, output_shape, wrap_array)
            )) {
        auto left_expr  = MshadowWrapper<devT,T,decltype(left)>::wrap(left,  device, output_shape, wrap_array);
        auto right_expr = MshadowWrapper<devT,T,decltype(right)>::wrap(right, device, output_shape, wrap_array);
        typedef typename functor_helper::BinaryExtractDType<decltype(left_expr), decltype(right_expr)>::value functor_dtype_t;
        return mshadow::expr::FIndexed<Functor<functor_dtype_t>>(left_expr, right_expr);
    }
};

template<template<class>class Functor, typename LeftT, typename RightT>
const int LazyBinaryIndexed<Functor, LeftT, RightT>::evaluation_dim = lazy::LazyBinaryEvaluationDim<LeftT, RightT>::value;

template<typename LeftT, typename RightT>
struct LazyOuter : public LazyFunction<LazyOuter<LeftT,RightT>, LeftT, RightT> {
    static const int evaluation_dim;
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

    template<int devT,typename T, int ndim>
    auto to_mshadow_expr(memory::Device device, const std::vector<int>& output_shape, const lazy::EvaluationSpec<devT, T, ndim>& wrap_array) const ->
            decltype(
                mshadow::expr::outer_product(
                     mshadow::expr::reshape(MshadowWrapper<devT,T,decltype(left)>::wrap(left, device, output_shape, wrap_array.template d<1>()), mshadow::Shape1(1)),
                     mshadow::expr::reshape(MshadowWrapper<devT,T,decltype(right)>::wrap(right, device, output_shape, wrap_array.template d<1>()), mshadow::Shape1(1))
                )
            ) {
        auto left_expr  = MshadowWrapper<devT,T,decltype(left)>::wrap(left,   device, left.bshape(), wrap_array.template d<1>());
        auto right_expr = MshadowWrapper<devT,T,decltype(right)>::wrap(right, device, right.bshape(), wrap_array.template d<1>());
        return mshadow::expr::outer_product(
            mshadow::expr::reshape(left_expr, mshadow::Shape1(std::abs(left.bshape()[0]))),
            mshadow::expr::reshape(right_expr, mshadow::Shape1(std::abs(right.bshape()[0])))
        );
    }
};

template<typename LeftT, typename RightT>
const int LazyOuter<LeftT, RightT>::evaluation_dim = 2;

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
