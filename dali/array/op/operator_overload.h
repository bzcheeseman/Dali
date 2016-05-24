#ifndef DALI_ARRAY_OP_OPERATOR_OVERLOAD_H
#define DALI_ARRAY_OP_OPERATOR_OVERLOAD_H

class Array;
class AssignableArray;

#define DALI_DECLARE_ARRAY_INTERACTION(SYMBOL)\
    AssignableArray operator SYMBOL (const Array& left, const Array& right);\

#define DALI_DECLARE_SCALAR_INTERACTION(SYMBOL)\
    AssignableArray operator SYMBOL (const Array& left, const double& right);\
    AssignableArray operator SYMBOL (const Array& left, const float& right);\
    AssignableArray operator SYMBOL (const Array& left, const int& right);\

DALI_DECLARE_ARRAY_INTERACTION(+);
DALI_DECLARE_ARRAY_INTERACTION(-);
DALI_DECLARE_ARRAY_INTERACTION(*);
DALI_DECLARE_ARRAY_INTERACTION(/);

DALI_DECLARE_SCALAR_INTERACTION(-);
DALI_DECLARE_SCALAR_INTERACTION(+);
DALI_DECLARE_SCALAR_INTERACTION(*);
DALI_DECLARE_SCALAR_INTERACTION(/);

#endif
