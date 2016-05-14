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

namespace internal {
    memory::AM UseOperator<OPERATOR_T_EQL>::access_mode = memory::AM_OVERWRITE;
    memory::AM UseOperator<OPERATOR_T_ADD>::access_mode = memory::AM_MUTABLE;
    memory::AM UseOperator<OPERATOR_T_SUB>::access_mode = memory::AM_MUTABLE;
    memory::AM UseOperator<OPERATOR_T_MUL>::access_mode = memory::AM_MUTABLE;
    memory::AM UseOperator<OPERATOR_T_DIV>::access_mode = memory::AM_MUTABLE;
}  // namespace internal
