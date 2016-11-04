#ifndef DALI_ARRAY_OP_OVERLOAD_NONLAZY_H
#define DALI_ARRAY_OP_OVERLOAD_NONLAZY_H

#include "dali/array/op2/expression/expression.h"

#define DALI_DECLARE_ARRAY_INTERACTION(SYMBOL)\
    Expression operator SYMBOL (const Array& left, const Array& right);\
    Expression operator SYMBOL (const Array& left, const Expression& right);\
    Expression operator SYMBOL (const Expression& left, const Array& right);\
    Expression operator SYMBOL (const Expression& left, const Expression& right);

#define DALI_DECLARE_SCALAR_INTERACTION(SYMBOL)\
    Expression operator SYMBOL (const Array& left, const double& right);\
    Expression operator SYMBOL (const Array& left, const float& right);\
    Expression operator SYMBOL (const Array& left, const int& right);\
    Expression operator SYMBOL (const Expression& left, const double& right);\
    Expression operator SYMBOL (const Expression& left, const float& right);\
    Expression operator SYMBOL (const Expression& left, const int& right);\
    Expression operator SYMBOL (const double& left, const Array& right);\
    Expression operator SYMBOL (const float& left, const Array& right);\
    Expression operator SYMBOL (const int& left, const Array& right);\
    Expression operator SYMBOL (const double& left, const Expression& right);\
    Expression operator SYMBOL (const float& left, const Expression& right);\
    Expression operator SYMBOL (const int& left, const Expression& right);

DALI_DECLARE_ARRAY_INTERACTION(+);
DALI_DECLARE_ARRAY_INTERACTION(-);
DALI_DECLARE_ARRAY_INTERACTION(*);
DALI_DECLARE_ARRAY_INTERACTION(/);

DALI_DECLARE_SCALAR_INTERACTION(-);
DALI_DECLARE_SCALAR_INTERACTION(+);
DALI_DECLARE_SCALAR_INTERACTION(*);
DALI_DECLARE_SCALAR_INTERACTION(/);

Expression operator-(const Array& in);
Expression operator-(const Expression& in);

#endif
