#include "common.h"

#include "dali/array/function/operator.h"
#include "dali/array/array.h"
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


#define DALI_DEFINE_ARRAYGATHER_INTERACTION_INPLACE(OPERATOR, OPERATOR_NAME)\
    ArrayGather& operator OPERATOR (ArrayGather& left, const Array& assignable) {\
        return left OPERATOR (Assignable<ArrayGather>)lazy::identity(assignable);\
    } \
    void operator OPERATOR (ArrayGather&& left, const Array& assignable) {\
        ArrayGather left_instance = left;\
        left_instance OPERATOR (Assignable<ArrayGather>)lazy::identity(assignable);\
    } \
    ArrayGather& operator OPERATOR (ArrayGather& left, const Assignable<Array>& assignable) {\
        Array self_as_array = left;\
        self_as_array OPERATOR assignable;\
        return (left = self_as_array);\
    }\
    void operator OPERATOR (ArrayGather&& left, const Assignable<Array>& assignable) {\
        ArrayGather left_instance = left;\
        Array self_as_array = left_instance;\
        self_as_array OPERATOR assignable;\
        left = self_as_array;\
    }\
    ArrayGather& operator OPERATOR(ArrayGather& left, const Assignable<ArrayGather>& assignable) {\
        assignable.assign_to(left, OPERATOR_NAME);\
        return left;\
    }\
    void operator OPERATOR(ArrayGather&& left, const Assignable<ArrayGather>& assignable) {\
        ArrayGather left_instance = left;\
        assignable.assign_to(left_instance, OPERATOR_NAME);\
    }\


#define DALI_DEFINE_ARRAYGATHER_SCALAR_INTERACTION_INPLACE(OPNAME, SYMBOL)\
    ArrayGather& operator SYMBOL (ArrayGather& left, const double right) {\
        return left = OPNAME (lazy::take(left.source, left.indices), right);\
    }\
    void operator SYMBOL (ArrayGather&& left, const double right) {\
        ArrayGather left_instance = left;\
        left_instance = OPNAME (lazy::take(left_instance.source, left_instance.indices), right);\
    }\
    ArrayGather& operator SYMBOL (ArrayGather& left, const float right) {\
        return left = OPNAME (lazy::take(left.source, left.indices), right);\
    }\
    void operator SYMBOL (ArrayGather&& left, const float right) {\
        ArrayGather left_instance = left;\
        left_instance = OPNAME (lazy::take(left_instance.source, left_instance.indices), right);\
    }\
    ArrayGather& operator SYMBOL (ArrayGather& left, const int right) {\
        return left = OPNAME (lazy::take(left.source, left.indices), right);\
    }\
    void operator SYMBOL (ArrayGather&& left, const int right) {\
        ArrayGather left_instance = left;\
        left_instance = OPNAME (lazy::take(left_instance.source, left_instance.indices), right);\
    }\

DALI_DEFINE_ARRAYGATHER_INTERACTION_INPLACE(+=, OPERATOR_T_ADD);
DALI_DEFINE_ARRAYGATHER_INTERACTION_INPLACE(-=, OPERATOR_T_SUB);
DALI_DEFINE_ARRAYGATHER_INTERACTION_INPLACE(*=, OPERATOR_T_MUL);
DALI_DEFINE_ARRAYGATHER_INTERACTION_INPLACE(/=, OPERATOR_T_DIV);

DALI_DEFINE_ARRAYGATHER_SCALAR_INTERACTION_INPLACE(lazy::sub, -=);
DALI_DEFINE_ARRAYGATHER_SCALAR_INTERACTION_INPLACE(lazy::add, +=);
DALI_DEFINE_ARRAYGATHER_SCALAR_INTERACTION_INPLACE(lazy::eltmul, *=);
DALI_DEFINE_ARRAYGATHER_SCALAR_INTERACTION_INPLACE(lazy::eltdiv, /=);
