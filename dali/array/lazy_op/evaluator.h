#ifndef DALI_ARRAY_LAZY_OP_EVALUATOR_H
#define DALI_ARRAY_LAZY_OP_EVALUATOR_H

#include "dali/array/function/function.h"
#include "dali/array/dtype.h"

template<template<class>class Functor, typename LeftT, typename RightT>
struct Binary;

template<typename Reducer>
struct UnfoldingReducer {
    typedef std::tuple<typename Reducer::outtype_t, typename Reducer::state_t> outtuple_t;

    template<template<class>class Functor, typename LeftT, typename RightT, typename... Args>
    static outtuple_t unfold_helper(const outtuple_t& state, const Binary<Functor, LeftT,RightT>& binary_expr, const Args&... args) {
        return unfold_helper(state, binary_expr.left, binary_expr.right, args...);
    }

    template<typename T, typename... Args>
    static outtuple_t unfold_helper(const outtuple_t& state, const T& arg, const Args&... args) {
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


template<int devT,typename T>
struct MshadowWrapper {
    static inline auto to_expr(const Array& array, memory::Device device) -> decltype(MArray<devT,T>(array, device).d1()) {
        return MArray<devT,T>(array, device).d1();
    }

    static inline T to_expr(const float& scalar, memory::Device device) { return (T)scalar; }

    static inline T to_expr(const double& scalar, memory::Device device) { return (T)scalar; }

    static inline T to_expr(const int& scalar, memory::Device device) { return (T)scalar; }

    template<template<class>class Functor, typename LeftT, typename RightT>
    static inline auto to_expr(const Binary<Functor,LeftT,RightT>& sth, memory::Device device) ->
            decltype(
                mshadow::expr::F<Functor<T>>(
                     MshadowWrapper<devT,T>::to_expr(sth.left, device),
                     MshadowWrapper<devT,T>::to_expr(sth.right, device)
                )
            ) {
        auto left_expr  = MshadowWrapper<devT,T>::to_expr(sth.left,  device);
        auto right_expr = MshadowWrapper<devT,T>::to_expr(sth.right, device);
        return mshadow::expr::F<Functor<T>>(left_expr, right_expr);
    }

    /////////////////////// MYSTERY WARNING ////////////////////////////////////
    //   It is a complete mystery to me why only the above function,          //
    //   gets succesfully mached to Binary<mul,Array,Binary<mul,Array,Array>> //
    //   Below I present N versions that do not work:                         //
    ////////////////////////////////////////////////////////////////////////////


    // template<typename Expr>
    // static inline auto to_expr(const RValueExp<Expr>& expr,
    //                            memory::Device device) ->
    //                                decltype(expr.self().template to_mshadow_expr<devT,T>(device)) {
    //     return expr.self().template to_mshadow_expr<devT,T>(device);
    // }

    // template<typename ExprT>
    // static inline auto to_expr(const ExprT& sth,
    //                            memory::Device device) ->
    //                                decltype(sth.template to_mshadow_expr<devT,T>(device)) {
    //     ELOG("hit binary");
    //     return sth.template to_mshadow_expr<devT,T>(device);
    // }

    // template<template<class>class Functor, typename LeftT, typename RightT>
    // static inline auto to_expr(const Binary<Functor,LeftT,RightT>& sth,
    //                            memory::Device device) ->
    //                                decltype(sth.template to_mshadow_expr<devT,T>(device)) {
    //     ELOG("hit binary");
    //     return sth.template to_mshadow_expr<devT,T>(device);
    // }
};


template<class LazyExpr>
struct Evaluator : public Function<Evaluator<LazyExpr>, Array, LazyExpr> {

    static std::vector<int> deduce_output_shape(const LazyExpr& expr) {
        return expr.shape();
    }

    static DType deduce_output_dtype(const LazyExpr& expr) {
        return expr.dtype();
    }

    static memory::Device deduce_output_device(const LazyExpr& expr) {
        return UnfoldingReducer<DeviceReducer>::reduce(expr);
    }

    static memory::Device deduce_computation_device(const Array& out, const LazyExpr& expr) {
        return UnfoldingReducer<DeviceReducer>::reduce(out, expr);
    }

    static DType deduce_computation_dtype(const Array& out, const LazyExpr& expr) {
        ASSERT2(out.dtype() == expr.dtype(),
            utils::MS() << "Output type (" << dtype_to_name(out.dtype())
                        << ") and expression type (" << dtype_to_name(expr.dtype()) << ") differ");
        return out.dtype();
    }


    template<int devT, typename T>
    void typed_eval(MArray<devT,T> out, const LazyExpr& expr) {
        out.d1(memory::AM_OVERWRITE) = MshadowWrapper<devT,T>::to_expr(expr, out.device);;
    }
};

#endif
