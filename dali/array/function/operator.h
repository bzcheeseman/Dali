#ifndef DALI_ARRAY_FUNCTION_OPERATOR_H
#define DALI_ARRAY_FUNCTION_OPERATOR_H

#include <string>
#include <iostream>
#include "dali/array/memory/access_modes.h"

// define different ways assignment between two expressions
// can happen:
enum OPERATOR_T {
    OPERATOR_T_EQL  = 0,/* =   */
    OPERATOR_T_ADD  = 1,/* +=  */
    OPERATOR_T_SUB  = 2,/* -=  */
    OPERATOR_T_MUL  = 3,/* *=  */
    OPERATOR_T_DIV  = 4,/* /=  */
    OPERATOR_T_LSE  = 5 /* <<= */
};

namespace internal {
    template<OPERATOR_T operator_t>
    struct UseOperator {
        // static_assert(false, "this method should not be used ever.");
    };

    #define DECLARE_OPERATOR_ASSIGN_HELPER(OPERATOR_T_SOMETHING, OPERATOR_LITERAL) \
        template<> \
        struct UseOperator<OPERATOR_T_SOMETHING> { \
            static memory::AM access_mode; \
            template<typename LeftType, typename RightType> \
            static inline auto apply(LeftType l, RightType r) -> decltype(l OPERATOR_LITERAL r) { \
                return l OPERATOR_LITERAL r; \
            } \
        };

    DECLARE_OPERATOR_ASSIGN_HELPER(OPERATOR_T_EQL, = );
    DECLARE_OPERATOR_ASSIGN_HELPER(OPERATOR_T_ADD, +=);
    DECLARE_OPERATOR_ASSIGN_HELPER(OPERATOR_T_SUB, -=);
    DECLARE_OPERATOR_ASSIGN_HELPER(OPERATOR_T_MUL, *=);
    DECLARE_OPERATOR_ASSIGN_HELPER(OPERATOR_T_DIV, /=);
    // TODO(jonathan): add a comment.
    DECLARE_OPERATOR_ASSIGN_HELPER(OPERATOR_T_LSE, +=);
};  // namespace internal


template<OPERATOR_T operator_t, int ndim, typename LeftType, typename RightType>
struct OperatorAssignHelper {
    static inline void assign_contiguous(LeftType& left, const RightType& right, bool collapse_leading=true) {
        internal::UseOperator<operator_t>::apply(
            left.template contiguous_d<ndim>(internal::UseOperator<operator_t>::access_mode, collapse_leading),
            right
        );
    }

    static inline void assign_noncontiguous(LeftType& left, const RightType& right, bool collapse_leading=true) {
        internal::UseOperator<operator_t>::apply(
            left.template d<ndim>(internal::UseOperator<operator_t>::access_mode, collapse_leading),
            right
        );
    }

    static inline void assign(LeftType& left, const RightType& right, bool collapse_leading=true) {
        if (left.array.contiguous_memory()) {
            assign_contiguous(left, right, collapse_leading);
        } else {
            assign_noncontiguous(left, right, collapse_leading);
        }
    }
};

template<OPERATOR_T operator_t, int ndim, typename LeftType, typename RightType>
void inline operator_assign(LeftType& left, const RightType& right, bool collapse_leading=true) {
    OperatorAssignHelper<operator_t,ndim,LeftType,RightType>::assign(left, right, collapse_leading);
}


template<OPERATOR_T operator_t, int ndim, typename LeftType, typename RightType>
void inline operator_assign_contiguous(LeftType& left, const RightType& right, bool collapse_leading=true) {
    OperatorAssignHelper<operator_t,ndim,LeftType,RightType>::assign_contiguous(left, right, collapse_leading);
}

std::string operator_to_name(const OPERATOR_T& operator_t);

#endif
