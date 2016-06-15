#ifndef DALI_ARRAY_OP_OVERLOAD_NONLAZY_H
#define DALI_ARRAY_OP_OVERLOAD_NONLAZY_H

class Array;
template<typename OutType>
class Assignable;

#define DALI_DECLARE_ARRAY_INTERACTION(SYMBOL)\
    Assignable<Array> operator SYMBOL (const Array& left, const Array& right);

#define DALI_DECLARE_SCALAR_INTERACTION(SYMBOL)\
    Assignable<Array> operator SYMBOL (const Array& left, const double& right);\
    Assignable<Array> operator SYMBOL (const Array& left, const float& right);\
    Assignable<Array> operator SYMBOL (const Array& left, const int& right);

DALI_DECLARE_ARRAY_INTERACTION(+);
DALI_DECLARE_ARRAY_INTERACTION(-);
DALI_DECLARE_ARRAY_INTERACTION(*);
DALI_DECLARE_ARRAY_INTERACTION(/);

DALI_DECLARE_SCALAR_INTERACTION(-);
DALI_DECLARE_SCALAR_INTERACTION(+);
DALI_DECLARE_SCALAR_INTERACTION(*);
DALI_DECLARE_SCALAR_INTERACTION(/);

#endif
