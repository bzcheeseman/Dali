#ifndef DALI_ARRAY_LAZY_EVALUATOR_H
#define DALI_ARRAY_LAZY_EVALUATOR_H

#include "dali/array/function/function.h"
#include "dali/array/dtype.h"

template<template<class>class Functor, typename LeftT, typename RightT>
struct LazyBinary;

template<template<class>class Functor, typename ExprT>
struct LazyUnary;

template<class Functor, typename ExprT>
struct LazyReducer;

template<typename Reducer>
struct UnfoldingReducer {
    typedef std::tuple<typename Reducer::outtype_t, typename Reducer::state_t> outtuple_t;

    template<template<class>class Functor, typename LeftT, typename RightT, typename... Args>
    static outtuple_t unfold_helper(
            const outtuple_t& state,
            const LazyBinary<Functor, LeftT,RightT>& binary_expr,
            const Args&... args) {
        return unfold_helper(state, binary_expr.left, binary_expr.right, args...);
    }

    template<template<class>class Functor, typename ExprT, typename... Args>
    static outtuple_t unfold_helper(
            const outtuple_t& state,
            const LazyUnary<Functor,ExprT>& elementwise_expr,
            const Args&... args) {
        return unfold_helper(state, elementwise_expr.expr, args...);
    }

    template<class Functor, typename ExprT, typename... Args>
    static outtuple_t unfold_helper(
            const outtuple_t& state,
            const LazyReducer<Functor,ExprT>& reducer_expr,
            const Args&... args) {
        return unfold_helper(state, reducer_expr.expr, args...);
    }

    template<typename T, typename... Args>
    static outtuple_t unfold_helper(const outtuple_t& state, const T& arg, const Args&... args) {
        static_assert(!std::is_base_of<LazyExpType,T>::value,
                "All Lazy expressions need to be explicitly expanded in UnfoldingReducer. Did you forget to cover and expression?");
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

template<int devT,typename T, typename ExprT>
struct MshadowWrapper {
    static inline auto to_expr(const ExprT& sth, memory::Device device) ->
            decltype(sth.template to_mshadow_expr<devT,T>(device)) {
        return sth.template to_mshadow_expr<devT,T>(device);
    }
};

template<int devT,typename T>
struct MshadowWrapper<devT,T,Array> {
    static inline auto to_expr(const Array& array, memory::Device device) ->
            decltype(TypedArray<devT,T>(array, device).d1()) {
        return TypedArray<devT,T>(array, device).d1();
    }
};

template<int devT,typename T>
struct MshadowWrapper<devT,T,float> {
    static inline T to_expr(const float& scalar, memory::Device device) { return (T)scalar; }
};

template<int devT,typename T>
struct MshadowWrapper<devT,T,double> {
    static inline T to_expr(const double& scalar, memory::Device device) { return (T)scalar; }
};

template<int devT,typename T>
struct MshadowWrapper<devT,T,int> {
    static inline T to_expr(const int& scalar, memory::Device device) { return (T)scalar; }
};

namespace debug {
    extern int evaluator_calls;
}

template<class LazyExpr>
struct Evaluator : public Function<Evaluator<LazyExpr>, Array, LazyExpr> {

    static std::vector<int> deduce_output_shape(const LazyExpr& expr) {
        return expr.shape();
    }

    static DType deduce_output_dtype(const LazyExpr& expr) {
        return expr.dtype();
    }

    static memory::Device deduce_output_device(const LazyExpr& expr) {
        auto res = UnfoldingReducer<DeviceReducer>::reduce(expr);
        return res;
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
    void typed_eval(TypedArray<devT,T> out, const LazyExpr& expr) {
        debug::evaluator_calls += 1;
        out.d1(memory::AM_OVERWRITE) =
                MshadowWrapper<devT,T,decltype(expr)>::to_expr(expr, out.device);
    }
};

#endif
