#ifndef DALI_MAT_MATH_LAZY_PLUCK_H
#define DALI_MAT_MATH_LAZY_PLUCK_H

#include "mshadow/tensor.h"
#include "mshadow/expr_engine-inl.h"
#include "dali/mat/math/LazyUtils.h"

namespace dali_expr {
    template<typename EType, typename DType>
    struct PluckExpression: public mshadow::expr::Exp<PluckExpression<EType, DType>,
                                      DType, mshadow::expr::type::kComplex> {
        const EType &exp;
        int idx;
        explicit PluckExpression(const EType &e, int _idx) : exp(e), idx(_idx) {}
    };
}

template<typename E, typename DType>
struct mshadow::expr::ExpInfo< dali_expr::PluckExpression<E, DType> > {
  static const int kDim = ExpInfo<E>::kDim - 1;
  static const int kDevMask = ExpInfo<E>::kDevMask;
};

template<typename SV, typename EType, typename DType>
struct mshadow::expr::ExpComplexEngine<SV,
                        typename extract_tensor_arguments<EType>::sub_tensor_t,
                        dali_expr::PluckExpression< EType, DType >,
                        DType > {
    inline static void Eval(typename extract_tensor_arguments<EType>::sub_tensor_t *dst,
                            const dali_expr::SoftmaxExpression< EType, DType > &exp) {

        *dst = exp.exp[exp.idx];
    }
};
/*
template<typename T, typename DType>
inline mshadow::expr::Plan<T, DType> mshadow::expr::MakePlan(const dali_expr::PluckExpression<T, DType> &e) {
    //return mshadow::expr::Plan<T, DType>(e.exp);
    return void
}*/


template<typename T, typename DType>
inline mshadow::expr::Plan<dali_expr::PluckExpression<T, DType>, DType>
MakePlan(const dali_expr::PluckExpression<T, DType> &e) {
    return mshadow::expr::Plan<dali_expr::PluckExpression<T, DType>, DType>(mshadow::expr::MakePlan(e.exp));
}


#endif
