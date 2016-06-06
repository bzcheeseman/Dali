#ifndef DALI_ARRAY_OP_H
#define DALI_ARRAY_OP_H

#include "dali/config.h"

#include "dali/array/op/other.h"
#include "dali/array/op/dot.h"
#include "dali/array/op/binary.h"
#include "dali/array/op/reducers.h"
#include "dali/array/op/reshape.h"
#include "dali/array/op/initializer.h"
#include "dali/array/op/unary.h"
#include "dali/array/op/softmax.h"
#include "dali/array/op/unary_scalar.h"
#include "dali/utils/print_utils.h"

#if EXISTS_AND_TRUE(DALI_USE_LAZY)
    #include "dali/array/lazy/binary.h"
    #include "dali/array/lazy/reducers.h"
    #include "dali/array/lazy/unary.h"
    #include "dali/array/lazy/reshape.h"
    #include "dali/array/function/lazy_evaluator.h"


    namespace lazy {
        static bool ops_loaded = true;
    }

    // Scalar templates need to be explicitly instantiated for every primitive
    // type we need to support. Otherwise C++ does not implicitly cast them.
    // this macro should be used N times with TYPE=[float,double,int,...]
    #define LAZY_BINARY_SCALAR_OPERATORS(OP,FNAME,TYPE) \
        template<typename T> \
        auto operator OP (const Exp<T>& left, const TYPE& right) -> decltype( FNAME (left.self(),right)) { \
            return FNAME (left.self(),right); \
        } \
        template<typename T> \
        auto operator OP (const TYPE& left, const Exp<T>& right) -> decltype( FNAME (left,right.self())) { \
            return FNAME (left,right.self()); \
        }

    #define LAZY_BINARY_OPERATOR(OP,FNAME) \
        template<typename T, typename T2> \
        auto operator OP(const Exp<T>& left, const Exp<T2>& right) -> decltype( FNAME (left.self(),right.self())) { \
            return FNAME (left.self(),right.self()); \
        } \
        LAZY_BINARY_SCALAR_OPERATORS(OP,FNAME,float) \
        LAZY_BINARY_SCALAR_OPERATORS(OP,FNAME,double) \
        LAZY_BINARY_SCALAR_OPERATORS(OP,FNAME,int)

        LAZY_BINARY_OPERATOR(+, lazy::add)
        LAZY_BINARY_OPERATOR(-, lazy::sub)
        LAZY_BINARY_OPERATOR(*, lazy::eltmul)
        LAZY_BINARY_OPERATOR(/, lazy::eltdiv)

    template<typename ExprT>
    auto operator-(const Exp<ExprT>& in) ->
            decltype(lazy::eltmul(-1,in.self())) {
        return lazy::eltmul(-1,in.self());
    }

    #define DALI_DECLARE_LAZY_INTERACTION_INPLACE(SYMBOL, SYMBOL_NAME) \
        template<typename ExprT> \
        Array& operator SYMBOL(Array& left, const LazyExp<ExprT>& right) { \
            return left SYMBOL lazy::EvalWithOperator<SYMBOL_NAME>::eval(right.self()); \
        } \

    DALI_DECLARE_LAZY_INTERACTION_INPLACE(+=, OPERATOR_T_ADD);
    DALI_DECLARE_LAZY_INTERACTION_INPLACE(-=, OPERATOR_T_SUB);
    DALI_DECLARE_LAZY_INTERACTION_INPLACE(*=, OPERATOR_T_MUL);
    DALI_DECLARE_LAZY_INTERACTION_INPLACE(/=, OPERATOR_T_DIV);
    DALI_DECLARE_LAZY_INTERACTION_INPLACE(<<=, OPERATOR_T_LSE);
#else
    namespace lazy {
        static bool ops_loaded = false;
    }

    #include "dali/array/op/operator_overload.h"

#endif

#endif
