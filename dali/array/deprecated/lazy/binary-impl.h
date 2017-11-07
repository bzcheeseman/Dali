// #include "dali/array/function/evaluation_dim.h"
// #include "dali/array/function/lazy_function.h"
// #include "dali/array/functor.h"

// namespace lazy {
//     template<typename LeftT, typename RightT>
//     struct LazyBinaryEvaluationDim {
//         static const int left_value = LazyEvaluationDim<LeftT>::value;
//         static const int right_value = LazyEvaluationDim<LeftT>::value;
//         static const bool values_disagree = left_value != right_value;
//         static const int value = (
//             (left_value == -1) ?
//                 right_value :
//                 ((right_value == -1) ?
//                      left_value :
//                      (values_disagree ?
//                         EVALUATION_DIM_ERROR :
//                         left_value
//                      )
//                 )
//         );
//     };
// }

// template<template<class>class Functor, typename LeftT, typename RightT>
// struct LazyBinary : public LazyFunction<LazyBinary<Functor,LeftT,RightT>, LeftT, RightT> {
//     static const int evaluation_dim;
//     LeftT  left;
//     RightT right;

//     LazyBinary(const LeftT& left_, const RightT& right_) :
//             LazyFunction<LazyBinary<Functor,LeftT,RightT>, LeftT, RightT>(left_, right_),
//             left(left_),
//             right(right_) {
//     }

//     template<int devT,typename T, int ndim>
//     auto to_mshadow_expr(memory::Device device, const std::vector<int>& output_shape, const lazy::EvaluationSpec<devT, T, ndim>& wrap_array) const ->
//             decltype(
//                 mshadow::expr::F<Functor<typename functor_helper::BinaryExtractDType<
//                     decltype(MshadowWrapper<devT,T,decltype(left)>::wrap(left, device, output_shape, wrap_array)),
//                     decltype(MshadowWrapper<devT,T,decltype(right)>::wrap(right, device, output_shape, wrap_array))
//                 >::value>
//             >(
//                      MshadowWrapper<devT,T,decltype(left)>::wrap(left, device, output_shape, wrap_array),
//                      MshadowWrapper<devT,T,decltype(right)>::wrap(right, device, output_shape, wrap_array)
//             )) {
//         auto left_expr  = MshadowWrapper<devT,T,decltype(left)>::wrap(left,  device, output_shape, wrap_array);
//         auto right_expr = MshadowWrapper<devT,T,decltype(right)>::wrap(right, device, output_shape, wrap_array);
//         typedef typename functor_helper::BinaryExtractDType<decltype(left_expr), decltype(right_expr)>::value functor_dtype_t;
//         return mshadow::expr::F<Functor<functor_dtype_t>>(left_expr, right_expr);
//     }
// };

// template<template<class>class Functor, typename Left, typename Right>
// struct LazyFunctionName<LazyBinary<Functor, Left, Right>> {
//     static std::string name;
// };

// template<template<class>class Functor, typename Left, typename Right>
// std::string LazyFunctionName<LazyBinary<Functor, Left, Right>>::name = "BinaryFunctor";

// template<template<class>class Functor, typename LeftT, typename RightT>
// const int LazyBinary<Functor, LeftT, RightT>::evaluation_dim = lazy::LazyBinaryEvaluationDim<LeftT, RightT>::value;

// namespace lazy {
//     template<template<class>class Functor, typename T1, typename T2>
//     LazyBinary<Functor,T1, T2> F(const T1& expr, const T2& expr2) {
//         return LazyBinary<Functor, T1, T2>(expr, expr2);
//     }

//     #define DALI_LAZY_IMPLEMENT_LAZYBINARY_EXPR(FUNCNAME, KERNELNAME)\
//         template <typename T1, typename T2>\
//         LazyBinary<KERNELNAME, T1, T2> FUNCNAME(const T1& a, const T2& b) {\
//             return LazyBinary<KERNELNAME, T1, T2>(a, b);\
//         }\

//     DALI_LAZY_IMPLEMENT_LAZYBINARY_EXPR(add, functor::add);
//     DALI_LAZY_IMPLEMENT_LAZYBINARY_EXPR(sub, functor::sub);
//     DALI_LAZY_IMPLEMENT_LAZYBINARY_EXPR(eltmul, functor::eltmul);
//     DALI_LAZY_IMPLEMENT_LAZYBINARY_EXPR(eltdiv, functor::eltdiv);
//     DALI_LAZY_IMPLEMENT_LAZYBINARY_EXPR(equals, functor::equals);
// }  // namespace lazy
