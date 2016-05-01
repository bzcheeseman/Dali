#ifndef DALI_ARRAY_FUNCTION_LAZY_EVALUATOR_H
#define DALI_ARRAY_FUNCTION_LAZY_EVALUATOR_H

#include "dali/array/dtype.h"
#include "dali/array/function/args/mshadow_wrapper.h"
#include "dali/array/function/args/reduce_over_lazy_expr.h"
#include "dali/array/function/function.h"
#include "dali/array/function/operator.h"

namespace debug {
    extern int lazy_evaluator_calls;
}

template<class LazyExpr>
struct LazyEvaluator : public Function<LazyEvaluator<LazyExpr>, Array, LazyExpr> {

    static std::vector<int> deduce_output_shape(const LazyExpr& expr) {
        return expr.shape();
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

        out.template d<LazyExpr::evaluation_dim>(memory::AM_OVERWRITE) =
                MshadowWrapper<devT,T,decltype(expr)>::wrap(expr, out.device);;
    }
};

#endif
