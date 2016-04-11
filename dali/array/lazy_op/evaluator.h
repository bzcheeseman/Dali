#ifndef DALI_ARRAY_LAZY_OP_EVALUATOR_H
#define DALI_ARRAY_LAZY_OP_EVALUATOR_H

#include "dali/array/array_function.h"

template<class LazyExpr>
struct Evaluator : public Function<Evaluator<LazyExpr>, Array, LazyExpr> {
    static std::vector<int> deduce_shape(LazyExpr i_could_not_care_less) {
        return std::vector<int>{12};
    }

    static DType deduce_dtype(LazyExpr i_could_not_care_less) {
        return DTYPE_FLOAT;
    }

    template<int devT, typename T>
    void typed_eval(MArray<devT,T> out, LazyExpr expr) {
        out.d1(memory::AM_OVERWRITE) =  expr.template eval<devT,T>();;
    }
};


template<int devT,typename T>
struct EvalLazy {
    template<typename ExprT>
    static inline auto eval(const ExprT& sth) -> decltype(sth.template eval<devT,T>()) {
        return sth.template eval<devT,T>();
    }

    static inline auto eval(const Array& array) -> decltype(MArray<devT,T>(array, memory::Device::cpu()).d1()) {
        return MArray<devT,T>(array, memory::Device::cpu()).d1();;
    }

    static inline float eval(const float& scalar) { return scalar; }

    static inline double eval(const double& scalar) { return scalar; }

    static inline int eval(const int& scalar) { return scalar; }
};


#endif
