#include "dali/array/function/lazy_function.h"
#include "dali/array/functor.h"

template<template<class>class Functor, typename ExprT>
struct LazyUnary : public LazyFunction<LazyUnary<Functor,ExprT>, ExprT> {
    ExprT expr;
    static const int evaluation_dim;

    LazyUnary(ExprT expr_) : LazyFunction<LazyUnary<Functor,ExprT>, ExprT>(expr_),
                             expr(expr_) {
    }

    template<int devT, typename T, int ndim>
    auto to_mshadow_expr(memory::Device device, const std::vector<int>& output_shape, const lazy::EvaluationSpec<devT, T, ndim>& wrap_array) const ->
            decltype(
                mshadow::expr::F<Functor<typename functor_helper::UnaryExtractDType<
                    decltype(MshadowWrapper<devT,T,ExprT>::wrap(expr, device, output_shape, wrap_array))>::value>
                >(MshadowWrapper<devT,T,ExprT>::wrap(expr, device, output_shape, wrap_array))
            ) {
        auto left_expr = MshadowWrapper<devT,T,ExprT>::wrap(expr, device, output_shape, wrap_array);
        typedef typename functor_helper::UnaryExtractDType<decltype(left_expr)>::value functor_dtype_t;
        return mshadow::expr::F<Functor<functor_dtype_t>>(left_expr);
    }
};

template<template<class>class Functor, typename ExprT>
const int LazyUnary<Functor, ExprT>::evaluation_dim = lazy::LazyEvaluationDim<ExprT>::value;

template<template<class>class Functor, typename ExprT>
struct LazyFunctionName<LazyUnary<Functor, ExprT>> {
    static std::string name;
};

template<template<class>class Functor, typename ExprT>
std::string LazyFunctionName<LazyUnary<Functor, ExprT>>::name = "UnaryFunctor";

namespace lazy {
    template<typename ExprT>
    LazyUnary<functor::identity,ExprT> identity(const Exp<ExprT>& expr) {
        return LazyUnary<functor::identity,ExprT>(expr.self());
    }
}  // namespace lazy
