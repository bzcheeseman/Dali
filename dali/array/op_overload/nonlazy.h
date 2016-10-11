#ifndef DALI_ARRAY_OP_OVERLOAD_NONLAZY_H
#define DALI_ARRAY_OP_OVERLOAD_NONLAZY_H

#include "dali/array/op2/operation.h"

#define DALI_DECLARE_ARRAY_INTERACTION(SYMBOL)\
    Operation operator SYMBOL (const Array& left, const Array& right);\
    Operation operator SYMBOL (const Array& left, const Operation& right);\
    Operation operator SYMBOL (const Operation& left, const Array& right);\
    Operation operator SYMBOL (const Operation& left, const Operation& right);

#define DALI_DECLARE_SCALAR_INTERACTION(SYMBOL)\
    Operation operator SYMBOL (const Array& left, const double& right);\
    Operation operator SYMBOL (const Array& left, const float& right);\
    Operation operator SYMBOL (const Array& left, const int& right);\
    Operation operator SYMBOL (const Operation& left, const double& right);\
    Operation operator SYMBOL (const Operation& left, const float& right);\
    Operation operator SYMBOL (const Operation& left, const int& right);\
    Operation operator SYMBOL (const double& left, const Array& right);\
    Operation operator SYMBOL (const float& left, const Array& right);\
    Operation operator SYMBOL (const int& left, const Array& right);\
    Operation operator SYMBOL (const double& left, const Operation& right);\
    Operation operator SYMBOL (const float& left, const Operation& right);\
    Operation operator SYMBOL (const int& left, const Operation& right);

DALI_DECLARE_ARRAY_INTERACTION(+);
DALI_DECLARE_ARRAY_INTERACTION(-);
DALI_DECLARE_ARRAY_INTERACTION(*);
DALI_DECLARE_ARRAY_INTERACTION(/);

DALI_DECLARE_SCALAR_INTERACTION(-);
DALI_DECLARE_SCALAR_INTERACTION(+);
DALI_DECLARE_SCALAR_INTERACTION(*);
DALI_DECLARE_SCALAR_INTERACTION(/);

Operation operator-(const Array& in);
Operation operator-(const Operation& in);

#endif
