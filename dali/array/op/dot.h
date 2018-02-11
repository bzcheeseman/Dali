#ifndef DALI_ARRAY_OP_DOT_H
#define DALI_ARRAY_OP_DOT_H

#include "dali/array/array.h"

namespace op {
    struct MatMul : public Expression {
        // matmul can be reshaped in-place
        MatMul(Array left, Array right, const std::vector<int>& shape);
        using Expression::copy;
        virtual expression_ptr copy() const override;
        memory::Device preferred_device() const override;
        virtual bool supports_operator(OPERATOR_T operator_t) const override;
        virtual expression_ptr _reshape(const std::vector<int>& new_shape,
                                        const Array* owner) const override;
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
