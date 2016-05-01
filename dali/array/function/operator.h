#ifndef DALI_ARRAY_FUNCTION_OPERATOR_H
#define DALI_ARRAY_FUNCTION_OPERATOR_H

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

#define DECLARE_OPERATOR_ASSIGN_HELPER_NDIM(NDIM, METHODNAME)\
    template<typename LeftType, typename RightType>\
    struct OperatorAssignHelper<OPERATOR_T_EQL, NDIM, LeftType, RightType> {\
        static inline void assign(LeftType& left, const RightType& right) {\
            left.METHODNAME (memory::AM_OVERWRITE) = right;\
        }\
    };\
    template<typename LeftType, typename RightType>\
    struct OperatorAssignHelper<OPERATOR_T_ADD, NDIM, LeftType, RightType> {\
        static inline void assign(LeftType& left, const RightType& right) {\
            left.METHODNAME (memory::AM_MUTABLE) += right;\
        }\
    };\
    template<typename LeftType, typename RightType>\
    struct OperatorAssignHelper<OPERATOR_T_SUB, NDIM, LeftType, RightType> {\
        static inline void assign(LeftType& left, const RightType& right) {\
            left.METHODNAME (memory::AM_MUTABLE) -= right;\
        }\
    };\
    template<typename LeftType, typename RightType>\
    struct OperatorAssignHelper<OPERATOR_T_DIV, NDIM, LeftType, RightType> {\
        static inline void assign(LeftType& left, const RightType& right) {\
            left.METHODNAME(memory::AM_MUTABLE) /= right;\
        }\
    };\
    template<typename LeftType, typename RightType>\
    struct OperatorAssignHelper<OPERATOR_T_MUL, NDIM, LeftType, RightType> {\
        static inline void assign(LeftType& left, const RightType& right) {\
            left.METHODNAME(memory::AM_MUTABLE) *= right;\
        }\
    }\


DECLARE_OPERATOR_ASSIGN_HELPER_NDIM(1, d1);
DECLARE_OPERATOR_ASSIGN_HELPER_NDIM(2, d2);
DECLARE_OPERATOR_ASSIGN_HELPER_NDIM(3, d3);
DECLARE_OPERATOR_ASSIGN_HELPER_NDIM(4, d4);

template<OPERATOR_T operator_t, int ndim, typename LeftType, typename RightType>
void inline operator_assign(LeftType& left, const RightType& right) {
    OperatorAssignHelper<operator_t,ndim,LeftType,RightType>::assign(left, right);
}


#endif
