#include "unary_scalar.h"
#include "dali/array/array.h"
#include "dali/array/lazy/binary.h"

namespace op {
    #define DALI_DEFINE_ARRAY_SCALAR_OP(FUNCTION_NAME, OPNAME)\
        AssignableArray FUNCTION_NAME(const Array& x, const double& other) {\
            return OPNAME(x, other);\
        }\
        AssignableArray FUNCTION_NAME(const Array& x, const float& other) {\
            return OPNAME(x, other);\
        }\
        AssignableArray FUNCTION_NAME(const Array& x, const int& other) {\
            return OPNAME(x, other);\
        }\

    DALI_DEFINE_ARRAY_SCALAR_OP(scalar_add, lazy::add);
    DALI_DEFINE_ARRAY_SCALAR_OP(scalar_sub, lazy::sub);
    DALI_DEFINE_ARRAY_SCALAR_OP(scalar_mul, lazy::eltmul);
    DALI_DEFINE_ARRAY_SCALAR_OP(scalar_div, lazy::eltdiv);
    DALI_DEFINE_ARRAY_SCALAR_OP(scalar_pow, lazy::pow);
};
