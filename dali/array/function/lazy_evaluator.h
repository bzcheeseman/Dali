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

template<class DestExpr, class SrcExpr>
struct LazyEvaluator : public Function<LazyEvaluator<DestExpr,SrcExpr>, DestExpr, SrcExpr> {
    static const int evaluation_dim =
            ((lazy::LazyEvaluationDim<SrcExpr>::value == lazy::EVALUATION_DIM_ANY) ?
            lazy::EVALUATION_DIM_DEFAULT :
            lazy::LazyEvaluationDim<SrcExpr>::value);

    static std::vector<int> deduce_output_bshape(const SrcExpr& expr) {
        return expr.bshape();
    }

    /* Deduce output type from input type InputT*/
    template<typename InputT>
    using out_expr_t = typename decltype(
        std::declval<SrcExpr>().to_mshadow_expr(
            std::declval<memory::Device>(),
            std::vector<int>(),
            std::declval<lazy::EvaluationSpec<memory::DEVICE_T_CPU, InputT, evaluation_dim>>()
        )
    )::exp_dtype_t;

    static DType deduce_inputs_dtype(const SrcExpr& expr) {
        return expr.dtype();
    }

    static DType deduce_output_dtype(const SrcExpr& expr) {
        auto input_dtype = deduce_inputs_dtype(expr);
        if (input_dtype == DTYPE_INT32) {
            return template_to_dtype<out_expr_t<int>>();
        } else if (input_dtype == DTYPE_FLOAT) {
            return template_to_dtype<out_expr_t<float>>();
        } else if (input_dtype == DTYPE_DOUBLE) {
            return template_to_dtype<out_expr_t<double>>();
        } else {
            ASSERT2(false, "received unknown dtype during output dtype deduction.");
        }
        return template_to_dtype<float>();
    }

    static DType deduce_computation_dtype(const DestExpr& out, const SrcExpr& expr) {
        ASSERT2(out.dtype() == deduce_output_dtype(expr),
            utils::MS() << "Output type (" << out.dtype()
                        << ") and expression type (" << expr.dtype() << ") differ");
        return deduce_inputs_dtype(expr);
    }

    static memory::Device deduce_output_device(const SrcExpr& expr) {
        return ReduceOverLazyExpr<DeviceReducer>::reduce(expr);
    }

    static memory::Device deduce_computation_device(const DestExpr& out, const SrcExpr& expr) {
        return ReduceOverLazyExpr<DeviceReducer>::reduce(out, expr);
    }

    template<OPERATOR_T operator_t, typename T, int devT, typename OutT>
    void typed_eval(TypedArray<devT,OutT> out, const SrcExpr& expr) {
        debug::lazy_evaluation_callback.notify(out.array);

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
            SrcExpr::collapse_leading
        );
    }

    template<OPERATOR_T operator_t, typename T, int devT, typename OutT, typename IndexT>
    void typed_eval(TypedArraySubtensor<devT,OutT,IndexT> out, const SrcExpr& expr) {
        debug::lazy_evaluation_callback.notify(out.source.array);

        operator_assign<operator_t, evaluation_dim>(
            out,
            MshadowWrapper<devT,T,decltype(expr)>::wrap(expr,
                                                        out.device,
                                                        out.shape,
                                                        lazy::EvaluationSpec<devT,T,evaluation_dim>()),
            SrcExpr::collapse_leading
        );
    }

    template<OPERATOR_T operator_t, typename T, int devT, typename OutT, typename IndexT>
    void typed_eval(TypedArrayGather<devT,OutT,IndexT> out, const SrcExpr& expr) {
        debug::lazy_evaluation_callback.notify(out.source.array);

        operator_assign<operator_t, evaluation_dim>(
            out,
            MshadowWrapper<devT,T,decltype(expr)>::wrap(expr,
                                                        out.device,
                                                        out.shape,
                                                        lazy::EvaluationSpec<devT,T,evaluation_dim>()),
            SrcExpr::collapse_leading
        );
    }
};

template<class DestExpr, class SrcExpr, typename T>
struct FunctionReturnType<LazyEvaluator<DestExpr,SrcExpr>, T> {
    typedef typename LazyEvaluator<DestExpr,SrcExpr>::template out_expr_t<T> value;
};

#include "dali/array/lazy/base_lazy_axis_reducer.h"

#endif
