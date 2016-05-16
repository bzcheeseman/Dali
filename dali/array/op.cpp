#define DALI_USE_LAZY 0
#include "op.h"

#define DALI_DEFINE_ARRAY_INTERACTION(OPNAME, SYMBOL)\
    AssignableArray operator SYMBOL (const Array& left, const Array& right) {\
        return OPNAME (left, right);\
    }\

#define DALI_DEFINE_ARRAY_INTERACTION_INPLACE(OPNAME, SYMBOL)\
    Array& operator SYMBOL (Array& left, const Array& right) {\
        left = OPNAME (left, right);\
        return left;\
    }\

#define DALI_DEFINE_SCALAR_INTERACTION(OPNAME, SYMBOL)\
    AssignableArray operator SYMBOL (const Array& left, const double& right) {\
        return OPNAME(left, right);\
    }\
    AssignableArray operator SYMBOL (const Array& left, const float& right) {\
        return OPNAME(left, right);\
    }\
    AssignableArray operator SYMBOL (const Array& left, const int& right) {\
        return OPNAME(left, right);\
    }\

#define DALI_DEFINE_SCALAR_INTERACTION_INPLACE(OPNAME, SYMBOL)\
    Array& operator SYMBOL (Array& left, const double& right) {\
        left = OPNAME(left, right);\
        return left;\
    }\
    Array& operator SYMBOL (Array& left, const float& right) {\
        left = OPNAME(left, right);\
        return left;\
    }\
    Array& operator SYMBOL (Array& left, const int& right) {\
        left = OPNAME(left, right);\
        return left;\
    }\

DALI_DEFINE_ARRAY_INTERACTION(op::add, +);
DALI_DEFINE_ARRAY_INTERACTION(op::sub, -);
DALI_DEFINE_ARRAY_INTERACTION(op::eltmul, *);
DALI_DEFINE_ARRAY_INTERACTION(op::eltdiv, /);
DALI_DEFINE_ARRAY_INTERACTION_INPLACE(op::add, +=);
DALI_DEFINE_ARRAY_INTERACTION_INPLACE(op::sub, -=);
DALI_DEFINE_ARRAY_INTERACTION_INPLACE(op::eltmul, *=);
DALI_DEFINE_ARRAY_INTERACTION_INPLACE(op::eltdiv, /=);

Array& operator<<=(Array& left, const Array& right) {
    left <<= op::identity(right);
    return left;
}

DALI_DEFINE_SCALAR_INTERACTION(op::scalar_sub, -);
DALI_DEFINE_SCALAR_INTERACTION(op::scalar_add, +);
DALI_DEFINE_SCALAR_INTERACTION(op::scalar_mul, *);
DALI_DEFINE_SCALAR_INTERACTION(op::scalar_div, /);
DALI_DEFINE_SCALAR_INTERACTION_INPLACE(op::scalar_sub, -=);
DALI_DEFINE_SCALAR_INTERACTION_INPLACE(op::scalar_add, +=);
DALI_DEFINE_SCALAR_INTERACTION_INPLACE(op::scalar_mul, *=);
DALI_DEFINE_SCALAR_INTERACTION_INPLACE(op::scalar_div, /=);
