// #ifndef DALI_ARRAY_OP_OVERLOAD_LAZY_H
// #define DALI_ARRAY_OP_OVERLOAD_LAZY_H

// #include "dali/array/function/expression.h"

// class Array;
// template<typename OutType>
// class Assignable;

// // Scalar templates need to be explicitly instantiated for every primitive
// // type we need to support. Otherwise C++ does not implicitly cast them.
// // this macro should be used N times with TYPE=[float,double,int,...]
// #define LAZY_BINARY_SCALAR_OPERATORS(OP,FNAME,TYPE) \
//     template<typename T> \
//     auto operator OP (const Exp<T>& left, const TYPE& right) -> decltype( FNAME (left.self(),right)) { \
//         return FNAME (left.self(),right); \
//     } \
//     template<typename T> \
//     auto operator OP (const TYPE& left, const Exp<T>& right) -> decltype( FNAME (left,right.self())) { \
//         return FNAME (left,right.self()); \
//     }

// #define LAZY_BINARY_OPERATOR(OP,FNAME) \
//     template<typename T, typename T2> \
//     auto operator OP(const Exp<T>& left, const Exp<T2>& right) -> decltype( FNAME (left.self(),right.self())) { \
//         return FNAME (left.self(),right.self()); \
//     } \
//     LAZY_BINARY_SCALAR_OPERATORS(OP,FNAME,float) \
//     LAZY_BINARY_SCALAR_OPERATORS(OP,FNAME,double) \
//     LAZY_BINARY_SCALAR_OPERATORS(OP,FNAME,int)


// #define DALI_DECLARE_LAZY_INTERACTION_INPLACE_CONTAINER(CONTAINER, SYMBOL, SYMBOL_NAME) \
//     template<typename ExprT> \
//     CONTAINER& operator SYMBOL(CONTAINER& left, const LazyExp<ExprT>& right) { \
//         return left SYMBOL lazy::EvalWithOperator<SYMBOL_NAME,CONTAINER>::eval(right.self()); \
//     }\
//     template<typename ExprT> \
//     void operator SYMBOL(CONTAINER&& left, const LazyExp<ExprT>& right) { \
//         left SYMBOL lazy::EvalWithOperator<SYMBOL_NAME,CONTAINER>::eval(right.self()); \
//     }

// #define DALI_DECLARE_LAZY_INTERACTION_INPLACE(SYMBOL, SYMBOL_NAME) \
//     DALI_DECLARE_LAZY_INTERACTION_INPLACE_CONTAINER(Array, SYMBOL, SYMBOL_NAME) \
//     DALI_DECLARE_LAZY_INTERACTION_INPLACE_CONTAINER(ArraySubtensor, SYMBOL, SYMBOL_NAME) \
//     DALI_DECLARE_LAZY_INTERACTION_INPLACE_CONTAINER(ArrayGather, SYMBOL, SYMBOL_NAME)

// LAZY_BINARY_OPERATOR(+, lazy::add)
// LAZY_BINARY_OPERATOR(-, lazy::sub)
// LAZY_BINARY_OPERATOR(*, lazy::eltmul)
// LAZY_BINARY_OPERATOR(/, lazy::eltdiv)

// template<typename ExprT>
// auto operator-(const Exp<ExprT>& in) ->
//         decltype(lazy::eltmul(-1,in.self())) {
//     return lazy::eltmul(-1,in.self());
// }

// DALI_DECLARE_LAZY_INTERACTION_INPLACE(+=, OPERATOR_T_ADD);
// DALI_DECLARE_LAZY_INTERACTION_INPLACE(-=, OPERATOR_T_SUB);
// DALI_DECLARE_LAZY_INTERACTION_INPLACE(*=, OPERATOR_T_MUL);
// DALI_DECLARE_LAZY_INTERACTION_INPLACE(/=, OPERATOR_T_DIV);
// DALI_DECLARE_LAZY_INTERACTION_INPLACE(<<=, OPERATOR_T_LSE);

// #endif
