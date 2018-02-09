#ifndef DALI_ARRAY_OP_DOT_H
#define DALI_ARRAY_OP_DOT_H

#include "dali/array/array.h"

namespace op {
    struct MatMul : public Expression {
        MatMul(Array left, Array right);
        using Expression::copy;
        virtual expression_ptr copy() const;
        memory::Device preferred_device() const;
        virtual bool supports_operator(OPERATOR_T operator_t) const;
    };
    Array matrix_multiply_with_reshape(const Array& a,
                                       const Array& b,
                                       const std::vector<int>& out_shape,
                                       const std::vector<int>& out_shape_2d);
    Array matmul(Array a, Array b);
    Array inner(const Array& a, const Array& b);
    Array tensordot(const Array& a, const Array& b, int axis);
    Array tensordot(const Array& a, const Array& b,
                    const std::vector<int>& a_reduce_axes,
                    const std::vector<int>& b_reduce_axes);
    Array matrix_vector_dot(const Array& a, const Array& b);
    Array dot(const Array& a, const Array& b);
}

#endif
