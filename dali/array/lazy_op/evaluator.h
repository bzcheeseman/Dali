#ifndef DALI_ARRAY_LAZY_OP_EVALUATOR_H
#define DALI_ARRAY_LAZY_OP_EVALUATOR_H

#include "dali/array/function/function.h"
#include "dali/array/dtype.h"

template<class LazyExpr>
struct Evaluator : public Function<Evaluator<LazyExpr>, Array, LazyExpr> {

    static std::vector<int> deduce_output_shape(const LazyExpr& expr) {
        return expr.shape();
    }

    static DType deduce_output_dtype(const LazyExpr& expr) {
        return expr.dtype();
    }

    // static memory::Device deduce_computation_device(const Array& out, const LazyExpr& expr) {
    //     return ReduceOverArgs<DeviceReducer>::reduce(out, args...);
    // }

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
        return MArray<devT,T>(array, memory::Device::cpu()).d1();;
    }

    static inline float to_expr(const float& scalar) { return scalar; }

    static inline double to_expr(const double& scalar) { return scalar; }

    static inline int to_expr(const int& scalar) { return scalar; }
};


#endif
