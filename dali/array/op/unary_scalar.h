#ifndef DALI_ARRAY_OP_UNARY_SCALAR_H
#define DALI_ARRAY_OP_UNARY_SCALAR_H

class Array;
class AssignableArray;

namespace op {
    #define DALI_DECLARE_ARRAY_SCALAR_OP(FUNCTION_NAME)\
        AssignableArray FUNCTION_NAME(const Array& x, const double& other);\
        AssignableArray FUNCTION_NAME(const Array& x, const float& other);\
        AssignableArray FUNCTION_NAME(const Array& x, const int& other);\

    DALI_DECLARE_ARRAY_SCALAR_OP(scalar_add);
    DALI_DECLARE_ARRAY_SCALAR_OP(scalar_sub);
    DALI_DECLARE_ARRAY_SCALAR_OP(scalar_mul);
    DALI_DECLARE_ARRAY_SCALAR_OP(scalar_div);
    DALI_DECLARE_ARRAY_SCALAR_OP(scalar_pow);

    #define DALI_DECLARE_ARRAY_SCALAR_OP_ARGS_REVERSED(FUNCTION_NAME)\
    	AssignableArray FUNCTION_NAME(const double& other, const Array& x);\
        AssignableArray FUNCTION_NAME(const float& other, const Array& x);\
        AssignableArray FUNCTION_NAME(const int& other, const Array& x);\

    DALI_DECLARE_ARRAY_SCALAR_OP_ARGS_REVERSED(scalar_sub);
    DALI_DECLARE_ARRAY_SCALAR_OP_ARGS_REVERSED(scalar_div);
    DALI_DECLARE_ARRAY_SCALAR_OP_ARGS_REVERSED(scalar_pow);
} // namespace op

#endif
