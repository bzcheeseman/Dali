#include "nonlazy.h"

#include "dali/array/array.h"
#include "dali/array/op2/binary.h"
#include "dali/array/op2/unary.h"

#define DALI_DEFINE_ARRAY_INTERACTION(OPNAME, SYMBOL)\
    expression::ExpressionGraph operator SYMBOL (const Array& left, const Array& right) {\
        return OPNAME (left, right);\
    }\
    expression::ExpressionGraph operator SYMBOL (const expression::ExpressionGraph& left, const Array& right) {\
        return OPNAME (left, right);\
    }\
    expression::ExpressionGraph operator SYMBOL (const Array& left, const expression::ExpressionGraph& right) {\
        return OPNAME (left, right);\
    }\
    expression::ExpressionGraph operator SYMBOL (const expression::ExpressionGraph& left, const expression::ExpressionGraph& right) {\
        return OPNAME (left, right);\
    }\

#define DALI_DEFINE_SCALAR_INTERACTION(OPNAME, SYMBOL)\
    expression::ExpressionGraph operator SYMBOL (const Array& left, const double& right) {\
        return OPNAME(left, right);\
    }\
    expression::ExpressionGraph operator SYMBOL (const Array& left, const float& right) {\
        return OPNAME(left, right);\
    }\
    expression::ExpressionGraph operator SYMBOL (const Array& left, const int& right) {\
        return OPNAME(left, right);\
    }\
    expression::ExpressionGraph operator SYMBOL (const expression::ExpressionGraph& left, const double& right) {\
        return OPNAME(left, right);\
    }\
    expression::ExpressionGraph operator SYMBOL (const expression::ExpressionGraph& left, const float& right) {\
        return OPNAME(left, right);\
    }\
    expression::ExpressionGraph operator SYMBOL (const expression::ExpressionGraph& left, const int& right) {\
        return OPNAME(left, right);\
    }\
    expression::ExpressionGraph operator SYMBOL (const double& left, const Array& right) {\
        return OPNAME(left, right);\
    }\
    expression::ExpressionGraph operator SYMBOL (const float& left, const Array& right) {\
        return OPNAME(left, right);\
    }\
    expression::ExpressionGraph operator SYMBOL (const int& left, const Array& right) {\
        return OPNAME(left, right);\
    }\
    expression::ExpressionGraph operator SYMBOL (const double& left, const expression::ExpressionGraph& right) {\
        return OPNAME(left, right);\
    }\
    expression::ExpressionGraph operator SYMBOL (const float& left, const expression::ExpressionGraph& right) {\
        return OPNAME(left, right);\
    }\
    expression::ExpressionGraph operator SYMBOL (const int& left, const expression::ExpressionGraph& right) {\
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

expression::ExpressionGraph operator-(const expression::ExpressionGraph& x) {
    return op::eltmul(-1, x);
}
expression::ExpressionGraph operator-(const Array& in) {
    return op::eltmul(-1, in);
}
