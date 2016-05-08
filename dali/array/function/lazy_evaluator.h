#ifndef DALI_ARRAY_FUNCTION_LAZY_EVALUATOR_H
#define DALI_ARRAY_FUNCTION_LAZY_EVALUATOR_H

#include "dali/array/dtype.h"
#include "dali/array/function/args/reduce_over_lazy_expr.h"
#include "dali/array/function/function.h"
#include "dali/array/function/operator.h"
#include "dali/array/function/typed_array.h"

namespace debug {
    extern int lazy_evaluator_calls;
}

////////////////////////////////////////////////////////////////////////////////
//                             MSHADOW_WRAPPER                                //
//                                   ---                                      //
//  This class would not be needed at all if we defined to_mshadow_expr       //
//  function on Array. The reason not to do that is to hide all mshadow usage //
//  in cpp files whereever possible.                                          //
////////////////////////////////////////////////////////////////////////////////


template<int devT,typename T, typename ExprT>
struct MshadowWrapper {
    static inline auto wrap(const ExprT& sth, memory::Device device, const std::vector<int>& output_shape) ->
            decltype(sth.template to_mshadow_expr<devT,T>(device, output_shape)) {
        return sth.template to_mshadow_expr<devT,T>(device, output_shape);
    }
};

template<int devT,typename T>
struct MshadowWrapper<devT,T,Array> {
    static inline auto wrap(const Array& array, memory::Device device, const std::vector<int>& output_shape) ->
            decltype(TypedArray<devT,T>(array, device, output_shape).d2()) {
        return TypedArray<devT,T>(array, device, output_shape).d2();
    }
};

template<int devT,typename T>
struct MshadowWrapper<devT,T,float> {
    static inline T wrap(const float& scalar, memory::Device device, const std::vector<int>& output_shape) { return (T)scalar; }
};

template<int devT,typename T>
struct MshadowWrapper<devT,T,double> {
    static inline T wrap(const double& scalar, memory::Device device, const std::vector<int>& output_shape) { return (T)scalar; }
};

template<int devT,typename T>
struct MshadowWrapper<devT,T,int> {
    static inline T wrap(const int& scalar, memory::Device device, const std::vector<int>& output_shape) { return (T)scalar; }
};



template<class LazyExpr>
struct LazyEvaluator : public Function<LazyEvaluator<LazyExpr>, Array, LazyExpr> {

    static std::vector<int> deduce_output_bshape(const LazyExpr& expr) {
        return expr.bshape();
    }

    static DType deduce_output_dtype(const LazyExpr& expr) {
        return expr.dtype();
    }

    static memory::Device deduce_output_device(const LazyExpr& expr) {
        auto res = ReduceOverLazyExpr<DeviceReducer>::reduce(expr);
        return res;
    }

    static memory::Device deduce_computation_device(const Array& out, const LazyExpr& expr) {
        return ReduceOverLazyExpr<DeviceReducer>::reduce(out, expr);
    }

    static DType deduce_computation_dtype(const Array& out, const LazyExpr& expr) {
        ASSERT2(out.dtype() == expr.dtype(),
            utils::MS() << "Output type (" << dtype_to_name(out.dtype())
                        << ") and expression type (" << dtype_to_name(expr.dtype()) << ") differ");
        return out.dtype();
    }

    template<OPERATOR_T operator_t, int devT, typename T>
    void typed_eval(TypedArray<devT,T> out, const LazyExpr& expr) {
        debug::lazy_evaluator_calls += 1;


        // out.array.shape() is passed to MshadowWrapper as final destination
        // shape this means that all the input arguments will be broadcasted
        // to fit out.array.shape(). Here we are assuming that out.array.shape()
        // is not broadcasted, so when the computation actually happens
        // the shape is already fully known every step of the way.

        operator_assign<operator_t, LazyExpr::evaluation_dim>(
            out,
            MshadowWrapper<devT,T,decltype(expr)>::wrap(expr, out.device, out.array.shape())
        );
    }
};

#endif
