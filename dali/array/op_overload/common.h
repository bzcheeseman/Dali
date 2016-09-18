#ifndef DALI_ARRAY_OP_OVERLOAD_COMMON_H
#define DALI_ARRAY_OP_OVERLOAD_COMMON_H

class Array;
class ArrayGather;
class ArraySubtensor;
template<typename OutType>
class Assignable;
class FusedOperation;

#define DALI_DECLARE_ARRAY_INTERACTION_INPLACE(SYMBOL)\
    Array& operator SYMBOL (Array&  left, const Assignable<Array>& right);\
    void operator SYMBOL (Array&& left, const Assignable<Array>& right);\
    Array& operator SYMBOL (Array&  left, const FusedOperation& right);\
    void operator SYMBOL (Array&& left, const FusedOperation& right);\
    Array& operator SYMBOL (Array&  left, const Array& right);\
    void operator SYMBOL (Array&& left, const Array& right);\


#define DALI_DECLARE_SCALAR_INTERACTION_INPLACE(SYMBOL, CONTAINER)\
    CONTAINER& operator SYMBOL (CONTAINER&  left, const double right);\
    void operator SYMBOL (CONTAINER&& left, const double right);\
    CONTAINER& operator SYMBOL (CONTAINER&  left, const float right);\
    void operator SYMBOL (CONTAINER&& left, const float right);\
    CONTAINER& operator SYMBOL (CONTAINER&  left, const int right);\
    void operator SYMBOL (CONTAINER&& left, const int right);\

// Array

DALI_DECLARE_ARRAY_INTERACTION_INPLACE(+= )
DALI_DECLARE_ARRAY_INTERACTION_INPLACE(-= )
DALI_DECLARE_ARRAY_INTERACTION_INPLACE(*= )
DALI_DECLARE_ARRAY_INTERACTION_INPLACE(/= )
DALI_DECLARE_ARRAY_INTERACTION_INPLACE(<<=)

DALI_DECLARE_SCALAR_INTERACTION_INPLACE(-=, Array)
DALI_DECLARE_SCALAR_INTERACTION_INPLACE(+=, Array)
DALI_DECLARE_SCALAR_INTERACTION_INPLACE(*=, Array)
DALI_DECLARE_SCALAR_INTERACTION_INPLACE(/=, Array)

// ArrayGather

#define DALI_DECLARE_CONTAINER_INTERACTION_INPLACE(SYMBOL, CONTAINER)\
    CONTAINER& operator SYMBOL (CONTAINER&  left, const Assignable<Array>& right);\
    void operator SYMBOL (CONTAINER&& left, const Assignable<Array>& right);\
    CONTAINER& operator SYMBOL (CONTAINER&  left, const Array& right);\
    void operator SYMBOL (CONTAINER&& left, const Array& right);\
    CONTAINER& operator SYMBOL (CONTAINER&  left, const Assignable<CONTAINER>& right);\
    void operator SYMBOL (CONTAINER&& left, const Assignable<CONTAINER>& right);\

DALI_DECLARE_CONTAINER_INTERACTION_INPLACE(+=, ArrayGather)
DALI_DECLARE_CONTAINER_INTERACTION_INPLACE(-=, ArrayGather)
DALI_DECLARE_CONTAINER_INTERACTION_INPLACE(*=, ArrayGather)
DALI_DECLARE_CONTAINER_INTERACTION_INPLACE(/=, ArrayGather)

DALI_DECLARE_SCALAR_INTERACTION_INPLACE(-=, ArrayGather)
DALI_DECLARE_SCALAR_INTERACTION_INPLACE(+=, ArrayGather)
DALI_DECLARE_SCALAR_INTERACTION_INPLACE(*=, ArrayGather)
DALI_DECLARE_SCALAR_INTERACTION_INPLACE(/=, ArrayGather)

DALI_DECLARE_CONTAINER_INTERACTION_INPLACE(+=, ArraySubtensor)
DALI_DECLARE_CONTAINER_INTERACTION_INPLACE(-=, ArraySubtensor)
DALI_DECLARE_CONTAINER_INTERACTION_INPLACE(*=, ArraySubtensor)
DALI_DECLARE_CONTAINER_INTERACTION_INPLACE(/=, ArraySubtensor)

DALI_DECLARE_SCALAR_INTERACTION_INPLACE(-=, ArraySubtensor)
DALI_DECLARE_SCALAR_INTERACTION_INPLACE(+=, ArraySubtensor)
DALI_DECLARE_SCALAR_INTERACTION_INPLACE(*=, ArraySubtensor)
DALI_DECLARE_SCALAR_INTERACTION_INPLACE(/=, ArraySubtensor)

#endif  // DALI_ARRAY_OP_OVERLOAD_COMMON_H
