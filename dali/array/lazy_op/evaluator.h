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

template<class LazyExpr>
struct Evaluator : public Function<Evaluator<LazyExpr>, Array, LazyExpr> {

    static std::vector<int> deduce_output_shape(const LazyExpr& expr) {
        return expr.shape();
    }

    static DType deduce_output_dtype(const LazyExpr& expr) {
        return expr.dtype();
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
    void typed_eval(MArray<devT,T> out, LazyExpr expr) {
        out.d1(memory::AM_OVERWRITE) =  expr.template to_mshadow_expr<devT,T>();;
    }
};


template<int devT,typename T>
struct MshadowWrapper {
    template<typename ExprT>
    static inline auto to_expr(const ExprT& sth) -> decltype(sth.template to_mshadow_expr<devT,T>()) {
        return sth.template to_mshadow_expr<devT,T>();
    }

    static inline auto to_expr(const Array& array) -> decltype(MArray<devT,T>(array, memory::Device::cpu()).d1()) {
        //TODO(szymon,jonathan): to_expr/to_mshadow_expr must receive target device to function correctly.
        return MArray<devT,T>(array, memory::Device::cpu()).d1();;
    }

    static inline float to_expr(const float& scalar) { return scalar; }

    static inline double to_expr(const double& scalar) { return scalar; }

    static inline int to_expr(const int& scalar) { return scalar; }
};


#endif
