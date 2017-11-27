#include "operator.h"

std::string operator_to_name(const OPERATOR_T& operator_t) {
    switch (operator_t) {
        case OPERATOR_T_EQL:
            return "=";
            break;
        case OPERATOR_T_ADD:
            return "+=";
            break;
        case OPERATOR_T_SUB:
            return "-=";
            break;
        case OPERATOR_T_MUL:
            return "*=";
            break;
        case OPERATOR_T_DIV:
            return "/=";
            break;
        case OPERATOR_T_LSE:
            return "<<=";
            break;
        default:
            return "unknown operator.";
    }
}
