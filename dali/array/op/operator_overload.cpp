#include "operator_overload.h"

#include "dali/array/array.h"
#include "dali/array/op/binary.h"
#include "dali/array/op/unary_scalar.h"

#define DALI_DEFINE_ARRAY_INTERACTION(OPNAME, SYMBOL)\
    AssignableArray operator SYMBOL (const Array& left, const Array& right) {\
        return OPNAME (left, right);\
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

DALI_DEFINE_ARRAY_INTERACTION(op::add, +);
DALI_DEFINE_ARRAY_INTERACTION(op::sub, -);
DALI_DEFINE_ARRAY_INTERACTION(op::eltmul, *);
DALI_DEFINE_ARRAY_INTERACTION(op::eltdiv, /);

DALI_DEFINE_SCALAR_INTERACTION(op::scalar_sub, -);
DALI_DEFINE_SCALAR_INTERACTION(op::scalar_add, +);
DALI_DEFINE_SCALAR_INTERACTION(op::scalar_mul, *);
DALI_DEFINE_SCALAR_INTERACTION(op::scalar_div, /);
