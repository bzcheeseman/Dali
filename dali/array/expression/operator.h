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

std::string operator_to_name(const OPERATOR_T& operator_t);
std::ostream& operator<<(std::ostream&, const OPERATOR_T&);

namespace std {
    template<>
    struct hash<OPERATOR_T> {
        std::size_t operator()(const OPERATOR_T&) const;
    };
}

#endif
