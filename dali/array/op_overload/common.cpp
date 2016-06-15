#include "common.h"

#include "dali/array/function/operator.h"
#include "dali/array/array.h"
#include "dali/array/op/unary.h"
#include "dali/array/op/unary_scalar.h"
#include "dali/array/op/binary.h"

#define DALI_DEFINE_ARRAY_INTERACTION_INPLACE(OPNAME, SYMBOL, OPERATOR_NAME)\
    Array& operator SYMBOL (Array& left, const Array& right) {\
        return left = OPNAME (left, right);\
    } \
    Array& operator SYMBOL (Array&& left, const Array& right) {\
        Array left_instance = left;\
        return left_instance = OPNAME (left_instance, right);\
    } \
    Array& operator SYMBOL (Array& left, const Assignable<Array>& assignable) {\
        assignable.assign_to(left, OPERATOR_NAME);\
        return left;\
    }\
    Array& operator SYMBOL (Array&& left, const Assignable<Array>& assignable) {\
        Array left_instance = left;\
        assignable.assign_to(left_instance, OPERATOR_NAME);\
        return left_instance;\
    }\


#define DALI_DEFINE_SCALAR_INTERACTION_INPLACE(OPNAME, SYMBOL)\
    Array& operator SYMBOL (Array& left, const double& right) {\
        return left = OPNAME (left, right);\
    }\
    Array& operator SYMBOL (Array&& left, const double& right) {\
        Array left_instance = left;\
        return left_instance = OPNAME (left_instance, right);\
    }\
    Array& operator SYMBOL (Array& left, const float& right) {\
        return left = OPNAME (left, right);\
    }\
    Array& operator SYMBOL (Array&& left, const float& right) {\
        Array left_instance = left;\
        return left_instance = OPNAME (left_instance, right);\
    }\
    Array& operator SYMBOL (Array& left, const int& right) {\
        return left = OPNAME (left, right);\
    }\
    Array& operator SYMBOL (Array&& left, const int& right) {\
        Array left_instance = left;\
        return left_instance = OPNAME (left_instance, right);\
    }\


DALI_DEFINE_ARRAY_INTERACTION_INPLACE(op::add,    +=, OPERATOR_T_ADD);
DALI_DEFINE_ARRAY_INTERACTION_INPLACE(op::sub,    -=, OPERATOR_T_SUB);
DALI_DEFINE_ARRAY_INTERACTION_INPLACE(op::eltmul, *=, OPERATOR_T_MUL);
DALI_DEFINE_ARRAY_INTERACTION_INPLACE(op::eltdiv, /=, OPERATOR_T_DIV);

Array& operator<<=(Array& left, const Array& right) {
    left <<= op::identity(right);
    return left;
}

Array& operator<<=(Array&& left, const Array& right) {
    Array left_instance = left;
    left_instance <<= op::identity(right);
    return left_instance;
}


Array& operator<<=(Array& left, const Assignable<Array>& assignable) {
    Array left_instance = left;
    assignable.assign_to(left_instance, OPERATOR_T_LSE);
    return left_instance;
}

DALI_DEFINE_SCALAR_INTERACTION_INPLACE(op::scalar_sub, -=);
DALI_DEFINE_SCALAR_INTERACTION_INPLACE(op::scalar_add, +=);
DALI_DEFINE_SCALAR_INTERACTION_INPLACE(op::scalar_mul, *=);
DALI_DEFINE_SCALAR_INTERACTION_INPLACE(op::scalar_div, /=);
