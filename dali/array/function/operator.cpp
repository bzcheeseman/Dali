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
        case OPERATOR_T_LSE:
            return "<<=";
        default:
            return "unknown operator";
    }
}

namespace std {
    std::size_t hash<OPERATOR_T>::operator()(const OPERATOR_T& operator_t) const {
        return operator_t;
    }
}

std::ostream& operator<<(std::ostream& stream, const OPERATOR_T& operator_t) {
    return stream << operator_to_name(operator_t);
}
