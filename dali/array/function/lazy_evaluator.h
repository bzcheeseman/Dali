#ifndef DALI_ARRAY_FUNCTION_LAZY_EVALUATOR_H
#define DALI_ARRAY_FUNCTION_LAZY_EVALUATOR_H

#include "dali/array/debug.h"
#include "dali/array/dtype.h"
#include "dali/array/function/evaluation_dim.h"
#include "dali/array/function/args/reduce_over_lazy_expr.h"
#include "dali/array/function/function.h"
#include "dali/array/function/lazy_function.h"
#include "dali/array/function/operator.h"
#include "dali/array/function/typed_array.h"


////////////////////////////////////////////////////////////////////////////////
//                             LAZY_EVALUATOR                                 //
////////////////////////////////////////////////////////////////////////////////

template<class LazyExpr>
struct LazyEvaluator : public Function<LazyEvaluator<LazyExpr>, Array, LazyExpr> {
    static const int evaluation_dim = (lazy::LazyEvaluationDim<LazyExpr>::value == lazy::EVALUATION_DIM_ANY) ?
                                       lazy::EVALUATION_DIM_DEFAULT : lazy::LazyEvaluationDim<LazyExpr>::value;

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

    template<OPERATOR_T operator_t, int devT, typename T, typename IndexT>
    void subtensor_assign_with_operator(TypedArraySubtensor<devT, T, IndexT> out, const LazyExpr& expr) {
        operator_assign<operator_t, evaluation_dim>(
            out,
            MshadowWrapper<devT,T,decltype(expr)>::wrap(expr,
                                                        out.device,
                                                        out.shape(),
                                                        lazy::EvaluationSpec<devT,T,evaluation_dim>()),
            LazyExpr::collapse_leading
        );
    }

    template<OPERATOR_T operator_t, int devT, typename T>
    void typed_eval(TypedArray<devT,T> out, const LazyExpr& expr) {
        debug::lazy_evaluation_callback.activate(out.array);

        // out.array.shape() is passed to MshadowWrapper as final destination
        // shape this means that all the input arguments will be broadcasted
        // to fit out.array.shape(). Here we are assuming that out.array.shape()
        // is not broadcasted, so when the computation actually happens
        // the shape is already fully known every step of the way.

        operator_assign<operator_t, evaluation_dim>(
            out,
            MshadowWrapper<devT,T,decltype(expr)>::wrap(expr,
                                                        out.device,
                                                        out.array.shape(),
                                                        lazy::EvaluationSpec<devT,T,evaluation_dim>()),
            LazyExpr::collapse_leading
        );
    }
};

#include "dali/array/lazy/base_lazy_axis_reducer.h"

#endif
