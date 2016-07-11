#ifndef DALI_ARRAY_OP_UNARY_SCALAR_H
#define DALI_ARRAY_OP_UNARY_SCALAR_H

class Array;
template<typename OutType>
class Assignable;

namespace op {
    #define DALI_DECLARE_ARRAY_SCALAR_OP(FUNCTION_NAME)\
        Assignable<Array> FUNCTION_NAME(const Array& x, const double& other);\
        Assignable<Array> FUNCTION_NAME(const Array& x, const float& other);\
        Assignable<Array> FUNCTION_NAME(const Array& x, const int& other);\

    DALI_DECLARE_ARRAY_SCALAR_OP(scalar_add);
    DALI_DECLARE_ARRAY_SCALAR_OP(scalar_sub);
    DALI_DECLARE_ARRAY_SCALAR_OP(scalar_mul);
    DALI_DECLARE_ARRAY_SCALAR_OP(scalar_div);
    DALI_DECLARE_ARRAY_SCALAR_OP(scalar_pow);

    #define DALI_DECLARE_ARRAY_SCALAR_OP_ARGS_REVERSED(FUNCTION_NAME)\
    	Assignable<Array> FUNCTION_NAME(const double& other, const Array& x);\
        Assignable<Array> FUNCTION_NAME(const float& other, const Array& x);\
        Assignable<Array> FUNCTION_NAME(const int& other, const Array& x);\

    DALI_DECLARE_ARRAY_SCALAR_OP_ARGS_REVERSED(scalar_add);
    DALI_DECLARE_ARRAY_SCALAR_OP_ARGS_REVERSED(scalar_sub);
    DALI_DECLARE_ARRAY_SCALAR_OP_ARGS_REVERSED(scalar_mul);
    DALI_DECLARE_ARRAY_SCALAR_OP_ARGS_REVERSED(scalar_div);
    DALI_DECLARE_ARRAY_SCALAR_OP_ARGS_REVERSED(scalar_pow);
} // namespace op

#endif
