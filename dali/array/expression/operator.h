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
        static_assert(
            (operator_t == OPERATOR_T_EQL ||
             operator_t == OPERATOR_T_ADD ||
             operator_t == OPERATOR_T_SUB ||
             operator_t == OPERATOR_T_MUL ||
             operator_t == OPERATOR_T_DIV ||
             operator_t == OPERATOR_T_LSE),
            "UseOperator can only be template-specialized using OPERATOR_T_EQL,"
            " OPERATOR_T_ADD, OPERATOR_T_SUB, OPERATOR_T_MUL, OPERATOR_T_DIV,"
            " or OPERATOR_T_LSE");
    };

    #define DECLARE_OPERATOR_ASSIGN_HELPER(OPERATOR_T_SOMETHING, OPERATOR_LITERAL) \
        template<> \
        struct UseOperator<OPERATOR_T_SOMETHING> { \
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
    // <<= is a syntactic sugar operator that performs
    // auto-reduction along the relevant dimensions when
    // the right operand is wider on dimensions that are
    // broadcasted on the left operand. If both sides
    // have equal dimensions this behaves like the regular
    // += operator.
    DECLARE_OPERATOR_ASSIGN_HELPER(OPERATOR_T_LSE, +=);

    template<OPERATOR_T operator_t>
    struct OperatorAM {
        template<typename LeftType>
        static memory::AM get(const LeftType& left) {
            return memory::AM_MUTABLE;
        }
    };

    template<>
    struct OperatorAM<OPERATOR_T_EQL> {
        template<typename LeftType>
        static memory::AM get(const LeftType& left) {
            if (left.spans_entire_memory()) {
                return memory::AM_OVERWRITE;
            } else {
                return memory::AM_MUTABLE;
            }
        }
    };

};  // namespace internal

std::string operator_to_name(const OPERATOR_T& operator_t);
std::ostream& operator<<(std::ostream&, const OPERATOR_T&);


namespace std {
    template<>
    struct hash<OPERATOR_T> {
        std::size_t operator()(const OPERATOR_T&) const;
    };
}


#endif
