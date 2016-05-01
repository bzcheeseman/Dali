#include "operator.h"

std::string operator_to_name(const OPERATOR_T& operator_t) {
    switch (operator_t) {
        case OPERATOR_T_EQL:
            return "=";
        case OPERATOR_T_ADD:
            return "+=";
        case OPERATOR_T_SUB:
            return "-=";
        case OPERATOR_T_DIV:
            return "/=";
        case OPERATOR_T_MUL:
            return "*=";
        default:
            return "unknown operator";
    }
}
