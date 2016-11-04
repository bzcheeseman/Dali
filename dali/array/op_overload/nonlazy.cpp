#include "nonlazy.h"

#include "dali/array/array.h"
#include "dali/array/op2/binary.h"
#include "dali/array/op2/unary.h"

#define DALI_DEFINE_ARRAY_INTERACTION(OPNAME, SYMBOL)\
    Expression operator SYMBOL (const Array& left, const Array& right) {\
        return OPNAME (left, right);\
    }\
    Expression operator SYMBOL (const Expression& left, const Array& right) {\
        return OPNAME (left, right);\
    }\
    Expression operator SYMBOL (const Array& left, const Expression& right) {\
        return OPNAME (left, right);\
    }\
    Expression operator SYMBOL (const Expression& left, const Expression& right) {\
        return OPNAME (left, right);\
    }\

#define DALI_DEFINE_SCALAR_INTERACTION(OPNAME, SYMBOL)\
    Expression operator SYMBOL (const Array& left, const double& right) {\
        return OPNAME(left, right);\
    }\
    Expression operator SYMBOL (const Array& left, const float& right) {\
        return OPNAME(left, right);\
    }\
    Expression operator SYMBOL (const Array& left, const int& right) {\
        return OPNAME(left, right);\
    }\
    Expression operator SYMBOL (const Expression& left, const double& right) {\
        return OPNAME(left, right);\
    }\
    Expression operator SYMBOL (const Expression& left, const float& right) {\
        return OPNAME(left, right);\
    }\
    Expression operator SYMBOL (const Expression& left, const int& right) {\
        return OPNAME(left, right);\
    }\
    Expression operator SYMBOL (const double& left, const Array& right) {\
        return OPNAME(left, right);\
    }\
    Expression operator SYMBOL (const float& left, const Array& right) {\
        return OPNAME(left, right);\
    }\
    Expression operator SYMBOL (const int& left, const Array& right) {\
        return OPNAME(left, right);\
    }\
    Expression operator SYMBOL (const double& left, const Expression& right) {\
        return OPNAME(left, right);\
    }\
    Expression operator SYMBOL (const float& left, const Expression& right) {\
        return OPNAME(left, right);\
    }\
    Expression operator SYMBOL (const int& left, const Expression& right) {\
        return OPNAME(left, right);\
    }\

DALI_DEFINE_ARRAY_INTERACTION(op::add, +);
DALI_DEFINE_ARRAY_INTERACTION(op::sub, -);
DALI_DEFINE_ARRAY_INTERACTION(op::eltmul, *);
DALI_DEFINE_ARRAY_INTERACTION(op::eltdiv, /);

DALI_DEFINE_SCALAR_INTERACTION(op::sub, -);
DALI_DEFINE_SCALAR_INTERACTION(op::add, +);
DALI_DEFINE_SCALAR_INTERACTION(op::eltmul, *);
DALI_DEFINE_SCALAR_INTERACTION(op::eltdiv, /);

Expression operator-(const Expression& x) {
    return op::eltmul(-1, x);
}
Expression operator-(const Array& in) {
    return op::eltmul(-1, in);
}
