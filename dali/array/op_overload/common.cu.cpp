#include "common.h"

#include "dali/array/function/operator.h"
#include "dali/array/array.h"
#include "dali/array/op2/operation.h"
#include "dali/array/lazy/unary.h"
#include "dali/array/lazy/binary.h"
#include "dali/array/lazy/reshape.h"
#include "dali/array/op/unary.h"
#include "dali/array/op/unary_scalar.h"
#include "dali/array/op/binary.h"

////////////////////////////////////////////////////////////
//                        ARRAY                           //
////////////////////////////////////////////////////////////

#define DALI_DEFINE_ARRAY_INTERACTION_INPLACE(OPNAME, SYMBOL, OPERATOR_NAME)\
    Array& operator SYMBOL (Array& left, const Array& right) {\
        return left = OPNAME (left, right);\
    } \
    void operator SYMBOL (Array&& left, const Array& right) {\
        Array left_instance = left;\
        left_instance = OPNAME (left_instance, right);\
    } \
    Array& operator SYMBOL (Array& left, const Assignable<Array>& assignable) {\
        assignable.assign_to(left, OPERATOR_NAME);\
        return left;\
    }\
    void operator SYMBOL (Array&& left, const Assignable<Array>& assignable) {\
        Array left_instance = left;\
        assignable.assign_to(left_instance, OPERATOR_NAME);\
    }\
    Array& operator SYMBOL (Array& left, const Operation& assignable) {\
        ((Assignable<Array>)assignable).assign_to(left, OPERATOR_NAME);\
        return left;\
    }\
    void operator SYMBOL (Array&& left, const Operation& assignable) {\
        Array left_instance = left;\
        ((Assignable<Array>)assignable).assign_to(left_instance, OPERATOR_NAME);\
    }\


#define DALI_DEFINE_SCALAR_INTERACTION_INPLACE(OPNAME, SYMBOL)\
    Array& operator SYMBOL (Array& left, const double right) {\
        return left = OPNAME (left, right);\
    }\
    void operator SYMBOL (Array&& left, const double right) {\
        Array left_instance = left;\
        left_instance = OPNAME (left_instance, right);\
    }\
    Array& operator SYMBOL (Array& left, const float right) {\
        return left = OPNAME (left, right);\
    }\
    void operator SYMBOL (Array&& left, const float right) {\
        Array left_instance = left;\
        left_instance = OPNAME (left_instance, right);\
    }\
    Array& operator SYMBOL (Array& left, const int right) {\
        return left = OPNAME (left, right);\
    }\
    void operator SYMBOL (Array&& left, const int right) {\
        Array left_instance = left;\
        left_instance = OPNAME (left_instance, right);\
    }\


DALI_DEFINE_ARRAY_INTERACTION_INPLACE(op::add,    +=, OPERATOR_T_ADD);
DALI_DEFINE_ARRAY_INTERACTION_INPLACE(op::sub,    -=, OPERATOR_T_SUB);
DALI_DEFINE_ARRAY_INTERACTION_INPLACE(op::eltmul, *=, OPERATOR_T_MUL);
DALI_DEFINE_ARRAY_INTERACTION_INPLACE(op::eltdiv, /=, OPERATOR_T_DIV);

Array& operator<<=(Array& left, const Array& right) {
    left <<= op::identity(right);
    return left;
}

void operator<<=(Array&& left, const Array& right) {
    Array left_instance = left;
    left_instance <<= op::identity(right);
}


Array& operator<<=(Array& left, const Assignable<Array>& assignable) {
    assignable.assign_to(left, OPERATOR_T_LSE);
    return left;
}

DALI_DEFINE_SCALAR_INTERACTION_INPLACE(op::scalar_sub, -=);
DALI_DEFINE_SCALAR_INTERACTION_INPLACE(op::scalar_add, +=);
DALI_DEFINE_SCALAR_INTERACTION_INPLACE(op::scalar_mul, *=);
DALI_DEFINE_SCALAR_INTERACTION_INPLACE(op::scalar_div, /=);

////////////////////////////////////////////////////////////
//                    ARRAY GATHER                        //
////////////////////////////////////////////////////////////


#define DALI_DEFINE_CONTAINER_INTERACTION_INPLACE(OPERATOR, OPERATOR_NAME, CONTAINER)\
    CONTAINER& operator OPERATOR (CONTAINER& left, const Array& assignable) {\
        return left OPERATOR (Assignable<CONTAINER>)lazy::identity(assignable);\
    } \
    void operator OPERATOR (CONTAINER&& left, const Array& assignable) {\
        CONTAINER left_instance = left;\
        left_instance OPERATOR (Assignable<CONTAINER>)lazy::identity(assignable);\
    } \
    CONTAINER& operator OPERATOR (CONTAINER& left, const Assignable<Array>& assignable) {\
        Array self_as_array = left;\
        self_as_array OPERATOR assignable;\
        return (left = self_as_array);\
    }\
    void operator OPERATOR (CONTAINER&& left, const Assignable<Array>& assignable) {\
        CONTAINER left_instance = left;\
        Array self_as_array = left_instance;\
        self_as_array OPERATOR assignable;\
        left = self_as_array;\
    }\
    CONTAINER& operator OPERATOR (CONTAINER& left, const Operation& assignable) {\
        ((Assignable<CONTAINER>)assignable).assign_to(left, OPERATOR_NAME);\
        return left;\
    }\
    void operator OPERATOR (CONTAINER&& left, const Operation& assignable) {\
        ((Assignable<CONTAINER>)assignable).assign_to(left, OPERATOR_NAME);\
    }\
    CONTAINER& operator OPERATOR(CONTAINER& left, const Assignable<CONTAINER>& assignable) {\
        assignable.assign_to(left, OPERATOR_NAME);\
        return left;\
    }\
    void operator OPERATOR(CONTAINER&& left, const Assignable<CONTAINER>& assignable) {\
        CONTAINER left_instance = left;\
        assignable.assign_to(left_instance, OPERATOR_NAME);\
    }\


#define DALI_DEFINE_CONTAINER_SCALAR_INTERACTION_INPLACE(OPNAME, SYMBOL, CONTAINER, EXTRACTOR)\
    CONTAINER& operator SYMBOL (CONTAINER& left, const double right) {\
        return left = OPNAME (EXTRACTOR(left.source, left.indices), right);\
    }\
    void operator SYMBOL (CONTAINER&& left, const double right) {\
        CONTAINER left_instance = left;\
        left_instance = OPNAME (EXTRACTOR(left_instance.source, left_instance.indices), right);\
    }\
    CONTAINER& operator SYMBOL (CONTAINER& left, const float right) {\
        return left = OPNAME (EXTRACTOR(left.source, left.indices), right);\
    }\
    void operator SYMBOL (CONTAINER&& left, const float right) {\
        CONTAINER left_instance = left;\
        left_instance = OPNAME (EXTRACTOR(left_instance.source, left_instance.indices), right);\
    }\
    CONTAINER& operator SYMBOL (CONTAINER& left, const int right) {\
        return left = OPNAME (EXTRACTOR(left.source, left.indices), right);\
    }\
    void operator SYMBOL (CONTAINER&& left, const int right) {\
        CONTAINER left_instance = left;\
        left_instance = OPNAME (EXTRACTOR(left_instance.source, left_instance.indices), right);\
    }\

DALI_DEFINE_CONTAINER_INTERACTION_INPLACE(+=, OPERATOR_T_ADD, ArrayGather);
DALI_DEFINE_CONTAINER_INTERACTION_INPLACE(-=, OPERATOR_T_SUB, ArrayGather);
DALI_DEFINE_CONTAINER_INTERACTION_INPLACE(*=, OPERATOR_T_MUL, ArrayGather);
DALI_DEFINE_CONTAINER_INTERACTION_INPLACE(/=, OPERATOR_T_DIV, ArrayGather);

DALI_DEFINE_CONTAINER_SCALAR_INTERACTION_INPLACE(lazy::sub, -=, ArrayGather, lazy::gather);
DALI_DEFINE_CONTAINER_SCALAR_INTERACTION_INPLACE(lazy::add, +=, ArrayGather, lazy::gather);
DALI_DEFINE_CONTAINER_SCALAR_INTERACTION_INPLACE(lazy::eltmul, *=, ArrayGather, lazy::gather);
DALI_DEFINE_CONTAINER_SCALAR_INTERACTION_INPLACE(lazy::eltdiv, /=, ArrayGather, lazy::gather);

DALI_DEFINE_CONTAINER_INTERACTION_INPLACE(+=, OPERATOR_T_ADD, ArraySubtensor);
DALI_DEFINE_CONTAINER_INTERACTION_INPLACE(-=, OPERATOR_T_SUB, ArraySubtensor);
DALI_DEFINE_CONTAINER_INTERACTION_INPLACE(*=, OPERATOR_T_MUL, ArraySubtensor);
DALI_DEFINE_CONTAINER_INTERACTION_INPLACE(/=, OPERATOR_T_DIV, ArraySubtensor);

DALI_DEFINE_CONTAINER_SCALAR_INTERACTION_INPLACE(lazy::sub, -=, ArraySubtensor, lazy::take_from_rows);
DALI_DEFINE_CONTAINER_SCALAR_INTERACTION_INPLACE(lazy::add, +=, ArraySubtensor, lazy::take_from_rows);
DALI_DEFINE_CONTAINER_SCALAR_INTERACTION_INPLACE(lazy::eltmul, *=, ArraySubtensor, lazy::take_from_rows);
DALI_DEFINE_CONTAINER_SCALAR_INTERACTION_INPLACE(lazy::eltdiv, /=, ArraySubtensor, lazy::take_from_rows);
