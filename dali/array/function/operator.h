#ifndef DALI_ARRAY_FUNCTION_OPERATOR_H
#define DALI_ARRAY_FUNCTION_OPERATOR_H

#include <string>
#include "dali/array/memory/access_modes.h"

// define different ways assignment between two expressions
// can happen:
enum OPERATOR_T {
    OPERATOR_T_EQL = 0,/* =  */
    OPERATOR_T_ADD = 1,/* += */
    OPERATOR_T_SUB = 2,/* -= */
    OPERATOR_T_MUL = 3,/* *= */
    OPERATOR_T_DIV = 4/* /= */
};

template<OPERATOR_T operator_t, int ndim, typename LeftType, typename RightType>
struct OperatorAssignHelper {
    static inline void assign(LeftType& left, const RightType& right);
};

#define DECLARE_OPERATOR_ASSIGN_HELPER_NDIM(OPERATOR_ENUM, OPERATOR_SYMBOL, MEMORY_ACCESS, NDIM, METHODNAME, CONTIGUOUS_METHODNAME)\
    template<typename LeftType, typename RightType>\
    struct OperatorAssignHelper<OPERATOR_ENUM, NDIM, LeftType, RightType> {\
        static inline void assign(LeftType& left, const RightType& right) {\
            if (left.array.contiguous_memory()) {\
                left.CONTIGUOUS_METHODNAME (MEMORY_ACCESS) OPERATOR_SYMBOL right;\
            } else {\
                left.METHODNAME (MEMORY_ACCESS) OPERATOR_SYMBOL right;\
            }\
        }\
    }\

// recursive macro to define many inlined operators that should be removed
#define DECLARE_ALL_OPERATOR_ASSIGN_HELPER_NDIM(NDIM, METHODNAME, CONTIGUOUS_METHODNAME)\
    DECLARE_OPERATOR_ASSIGN_HELPER_NDIM(OPERATOR_T_EQL, =,  memory::AM_OVERWRITE, NDIM, METHODNAME, CONTIGUOUS_METHODNAME);\
    DECLARE_OPERATOR_ASSIGN_HELPER_NDIM(OPERATOR_T_ADD, +=, memory::AM_MUTABLE,   NDIM, METHODNAME, CONTIGUOUS_METHODNAME);\
    DECLARE_OPERATOR_ASSIGN_HELPER_NDIM(OPERATOR_T_SUB, -=, memory::AM_MUTABLE,   NDIM, METHODNAME, CONTIGUOUS_METHODNAME);\
    DECLARE_OPERATOR_ASSIGN_HELPER_NDIM(OPERATOR_T_MUL, *=, memory::AM_MUTABLE,   NDIM, METHODNAME, CONTIGUOUS_METHODNAME);\
    DECLARE_OPERATOR_ASSIGN_HELPER_NDIM(OPERATOR_T_DIV, /=, memory::AM_MUTABLE,   NDIM, METHODNAME, CONTIGUOUS_METHODNAME);\

DECLARE_ALL_OPERATOR_ASSIGN_HELPER_NDIM(1, d1, contiguous_d1);
DECLARE_ALL_OPERATOR_ASSIGN_HELPER_NDIM(2, d2, contiguous_d2);

template<OPERATOR_T operator_t, int ndim, typename LeftType, typename RightType>
void inline operator_assign(LeftType& left, const RightType& right) {
    OperatorAssignHelper<operator_t,ndim,LeftType,RightType>::assign(left, right);
}

std::string operator_to_name(const OPERATOR_T& operator_t);

#endif
