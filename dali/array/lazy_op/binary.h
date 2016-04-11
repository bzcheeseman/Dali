#ifndef DALI_ARRAY_LAZY_OP_BINARY_OP
#define DALI_ARRAY_LAZY_OP_BINARY_OP

#include <tuple>

#include "dali/array/array_function.h"
#include "dali/array/TensorOps.h"

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


template<template<class>class Functor, typename LeftT, typename RightT>
struct Binary {
    typedef Binary<Functor,LeftT,RightT> self_t;
    LeftT  left;
    RightT right;

    Binary(const LeftT& _left, const RightT& _right) :
            left(_left), right(_right) {
    }

    template<int devT, typename T>
    inline auto eval() -> decltype(
                              mshadow::expr::F<Functor<T>>(
                                   EvalLazy<devT,T>::eval(left),
                                   EvalLazy<devT,T>::eval(right)
                              )
                          ) {
        auto left_expr  = EvalLazy<devT,T>::eval(left);
        auto right_expr = EvalLazy<devT,T>::eval(right);

        return mshadow::expr::F<Functor<T>>(left_expr, right_expr);

    }

    operator AssignableArray() const {
        return Evaluator<self_t>::run(*this);
    }
};

template <typename T, typename T2>
Binary<TensorOps::op::mul, T, T2> lazy_mul(T a, T2 b) {
    return Binary<TensorOps::op::mul, T, T2>(a, b);
}


#endif
