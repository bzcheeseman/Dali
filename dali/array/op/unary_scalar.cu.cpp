#include "unary_scalar.h"
#include "dali/array/array.h"
#include "dali/array/lazy/binary.h"

namespace op {
    #define DALI_DEFINE_ARRAY_SCALAR_OP(FUNCTION_NAME, OPNAME)\
        Assignable<Array> FUNCTION_NAME(const Array& x, const double& other) {\
            return OPNAME(x, other);\
        }\
        Assignable<Array> FUNCTION_NAME(const Array& x, const float& other) {\
            return OPNAME(x, other);\
        }\
        Assignable<Array> FUNCTION_NAME(const Array& x, const int& other) {\
            return OPNAME(x, other);\
        }\

    DALI_DEFINE_ARRAY_SCALAR_OP(scalar_add, lazy::add);
    DALI_DEFINE_ARRAY_SCALAR_OP(scalar_sub, lazy::sub);
    DALI_DEFINE_ARRAY_SCALAR_OP(scalar_mul, lazy::eltmul);
    DALI_DEFINE_ARRAY_SCALAR_OP(scalar_div, lazy::eltdiv);
    DALI_DEFINE_ARRAY_SCALAR_OP(scalar_pow, lazy::pow);

    #define DALI_DEFINE_ARRAY_SCALAR_OP_ARGS_REVERSED(FUNCTION_NAME, OPNAME)\
        Assignable<Array> FUNCTION_NAME(const double& other, const Array& x) {\
            return OPNAME(other, x);\
        }\
        Assignable<Array> FUNCTION_NAME(const float& other, const Array& x) {\
            return OPNAME(other, x);\
        }\
        Assignable<Array> FUNCTION_NAME(const int& other, const Array& x) {\
            return OPNAME(other, x);\
        }\

    DALI_DEFINE_ARRAY_SCALAR_OP_ARGS_REVERSED(scalar_add, lazy::add);
    DALI_DEFINE_ARRAY_SCALAR_OP_ARGS_REVERSED(scalar_sub, lazy::sub);
    DALI_DEFINE_ARRAY_SCALAR_OP_ARGS_REVERSED(scalar_mul, lazy::eltmul);
    DALI_DEFINE_ARRAY_SCALAR_OP_ARGS_REVERSED(scalar_div, lazy::eltdiv);
    DALI_DEFINE_ARRAY_SCALAR_OP_ARGS_REVERSED(scalar_pow, lazy::pow);

};
