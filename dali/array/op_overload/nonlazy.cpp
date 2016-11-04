#include "nonlazy.h"

#include "dali/array/array.h"
#include "dali/array/op2/binary.h"
#include "dali/array/op2/unary.h"

#define DALI_DEFINE_ARRAY_INTERACTION(OPNAME, SYMBOL)\
    expression::Expression operator SYMBOL (const Array& left, const Array& right) {\
        return OPNAME (left, right);\
    }\
    expression::Expression operator SYMBOL (const expression::Expression& left, const Array& right) {\
        return OPNAME (left, right);\
    }\
    expression::Expression operator SYMBOL (const Array& left, const expression::Expression& right) {\
        return OPNAME (left, right);\
    }\
    expression::Expression operator SYMBOL (const expression::Expression& left, const expression::Expression& right) {\
        return OPNAME (left, right);\
    }\

#define DALI_DEFINE_SCALAR_INTERACTION(OPNAME, SYMBOL)\
    expression::Expression operator SYMBOL (const Array& left, const double& right) {\
        return OPNAME(left, right);\
    }\
    expression::Expression operator SYMBOL (const Array& left, const float& right) {\
        return OPNAME(left, right);\
    }\
    expression::Expression operator SYMBOL (const Array& left, const int& right) {\
        return OPNAME(left, right);\
    }\
    expression::Expression operator SYMBOL (const expression::Expression& left, const double& right) {\
        return OPNAME(left, right);\
    }\
    expression::Expression operator SYMBOL (const expression::Expression& left, const float& right) {\
        return OPNAME(left, right);\
    }\
    expression::Expression operator SYMBOL (const expression::Expression& left, const int& right) {\
        return OPNAME(left, right);\
    }\
    expression::Expression operator SYMBOL (const double& left, const Array& right) {\
        return OPNAME(left, right);\
    }\
    expression::Expression operator SYMBOL (const float& left, const Array& right) {\
        return OPNAME(left, right);\
    }\
    expression::Expression operator SYMBOL (const int& left, const Array& right) {\
        return OPNAME(left, right);\
    }\
    expression::Expression operator SYMBOL (const double& left, const expression::Expression& right) {\
        return OPNAME(left, right);\
    }\
    expression::Expression operator SYMBOL (const float& left, const expression::Expression& right) {\
        return OPNAME(left, right);\
    }\
    expression::Expression operator SYMBOL (const int& left, const expression::Expression& right) {\
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

expression::Expression operator-(const expression::Expression& x) {
    return op::eltmul(-1, x);
}
expression::Expression operator-(const Array& in) {
    return op::eltmul(-1, in);
}
