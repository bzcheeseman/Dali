#ifndef DALI_ARRAY_OP_OVERLOAD_COMMON_H
#define DALI_ARRAY_OP_OVERLOAD_COMMON_H

class Array;
template<typename OutType>
class Assignable;

#define DALI_DECLARE_ARRAY_INTERACTION_INPLACE(SYMBOL)\
    Array& operator SYMBOL (Array&  left, const Assignable<Array>& right);\
    Array& operator SYMBOL (Array&& left, const Assignable<Array>& right);\
    Array& operator SYMBOL (Array&  left, const Array& right);\
    Array& operator SYMBOL (Array&& left, const Array& right);\


#define DALI_DECLARE_ARRAY_SCALAR_INTERACTION_INPLACE(SYMBOL)\
    Array& operator SYMBOL (Array&  left, const double& right);\
    Array& operator SYMBOL (Array&& left, const double& right);\
    Array& operator SYMBOL (Array&  left, const float& right);\
    Array& operator SYMBOL (Array&& left, const float& right);\
    Array& operator SYMBOL (Array&  left, const int& right);\
    Array& operator SYMBOL (Array&& left, const int& right);\


DALI_DECLARE_ARRAY_INTERACTION_INPLACE(+=)
DALI_DECLARE_ARRAY_INTERACTION_INPLACE(-=)
DALI_DECLARE_ARRAY_INTERACTION_INPLACE(*=)
DALI_DECLARE_ARRAY_INTERACTION_INPLACE(/=)
DALI_DECLARE_ARRAY_INTERACTION_INPLACE(<<=)

DALI_DECLARE_ARRAY_SCALAR_INTERACTION_INPLACE(-=)
DALI_DECLARE_ARRAY_SCALAR_INTERACTION_INPLACE(+=)
DALI_DECLARE_ARRAY_SCALAR_INTERACTION_INPLACE(*=)
DALI_DECLARE_ARRAY_SCALAR_INTERACTION_INPLACE(/=)

#endif  // DALI_ARRAY_OP_OVERLOAD_COMMON_H
