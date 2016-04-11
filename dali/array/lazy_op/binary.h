#ifndef DALI_ARRAY_LAZY_OP_BINARY_H
#define DALI_ARRAY_LAZY_OP_BINARY_H

#include "dali/array/lazy_op/evaluator.h"
#include "dali/array/TensorFunctions.h"

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


namespace lazy {
    template <typename T, typename T2>
    Binary<TensorOps::op::add, T, T2> add(T a, T2 b) {
        return Binary<TensorOps::op::add, T, T2>(a, b);
    }
    template <typename T, typename T2>
    Binary<TensorOps::op::sub, T, T2> sub(T a, T2 b) {
        return Binary<TensorOps::op::sub, T, T2>(a, b);
    }
    template <typename T, typename T2>
    Binary<TensorOps::op::eltmul, T, T2> eltmul(T a, T2 b) {
        return Binary<TensorOps::op::eltmul, T, T2>(a, b);
    }
    template <typename T, typename T2>
    Binary<TensorOps::op::eltdiv, T, T2> eltdiv(T a, T2 b) {
        return Binary<TensorOps::op::eltdiv, T, T2>(a, b);
    }
}


#endif
