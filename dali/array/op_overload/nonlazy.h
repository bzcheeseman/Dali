#ifndef DALI_ARRAY_OP_OVERLOAD_NONLAZY_H
#define DALI_ARRAY_OP_OVERLOAD_NONLAZY_H

#include "dali/array/op2/expression/expression.h"

#define DALI_DECLARE_ARRAY_INTERACTION(SYMBOL)\
    expression::Expression operator SYMBOL (const Array& left, const Array& right);\
    expression::Expression operator SYMBOL (const Array& left, const expression::Expression& right);\
    expression::Expression operator SYMBOL (const expression::Expression& left, const Array& right);\
    expression::Expression operator SYMBOL (const expression::Expression& left, const expression::Expression& right);

#define DALI_DECLARE_SCALAR_INTERACTION(SYMBOL)\
    expression::Expression operator SYMBOL (const Array& left, const double& right);\
    expression::Expression operator SYMBOL (const Array& left, const float& right);\
    expression::Expression operator SYMBOL (const Array& left, const int& right);\
    expression::Expression operator SYMBOL (const expression::Expression& left, const double& right);\
    expression::Expression operator SYMBOL (const expression::Expression& left, const float& right);\
    expression::Expression operator SYMBOL (const expression::Expression& left, const int& right);\
    expression::Expression operator SYMBOL (const double& left, const Array& right);\
    expression::Expression operator SYMBOL (const float& left, const Array& right);\
    expression::Expression operator SYMBOL (const int& left, const Array& right);\
    expression::Expression operator SYMBOL (const double& left, const expression::Expression& right);\
    expression::Expression operator SYMBOL (const float& left, const expression::Expression& right);\
    expression::Expression operator SYMBOL (const int& left, const expression::Expression& right);

DALI_DECLARE_ARRAY_INTERACTION(+);
DALI_DECLARE_ARRAY_INTERACTION(-);
DALI_DECLARE_ARRAY_INTERACTION(*);
DALI_DECLARE_ARRAY_INTERACTION(/);

DALI_DECLARE_SCALAR_INTERACTION(-);
DALI_DECLARE_SCALAR_INTERACTION(+);
DALI_DECLARE_SCALAR_INTERACTION(*);
DALI_DECLARE_SCALAR_INTERACTION(/);

expression::Expression operator-(const Array& in);
expression::Expression operator-(const expression::Expression& in);

#endif
