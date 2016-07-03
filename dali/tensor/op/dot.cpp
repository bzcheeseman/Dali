#include "dot.h"

#include "dali/array/op/tensordot_as_dot.h"
#include "dali/tensor/op/operators.h"
#include "dali/tensor/tape.h"
#include "dali/tensor/tensor_macros.h"
#include "dali/array/op.h"

Tensor matrixdot_with_custom_shape(const Tensor& a,
                                   const Tensor& b,
                                   std::vector<int> new_shape) {
    auto out = Tensor::empty(new_shape, a.dtype(), a.preferred_device());
    out.w = op::matrixdot(a.w, b.w);

    if (graph::backprop_enabled())
        graph::emplace_back([a, b, out]() mutable {
            MAYBE_GRAD(a) <<= op::matrixdot(out.dw, b.w.transpose());
            MAYBE_GRAD(b) <<= op::matrixdot(a.w.transpose(), out.dw);
        });
    return out;
}

namespace op {
    template<>
    Tensor matrix_multiply_with_reshape(const Tensor& a,
                                        const Tensor& b,
                                        const std::vector<int>& out_shape,
                                        const std::vector<int>& out_shape_2d) {
        return matrixdot_with_custom_shape(a, b, out_shape_2d).reshape(out_shape);
    }
}  // namespace op

namespace tensor_ops {

    Tensor dot(const Tensor& a, const Tensor& b) {
        auto a_ndim = a.ndim();
        auto b_ndim = b.ndim();

        if (a_ndim == 0 || b_ndim == 0) {
            if (a_ndim == 0) {
                return a.broadcast_scalar_to_ndim(b_ndim) * b;
            } else {
                return a * b.broadcast_scalar_to_ndim(a_ndim);
            }
        } else if (a_ndim > 2 || b_ndim > 2) {
            // a is reduced over the last dimension
            // b is reduced over second to last dimension if it exists,
            // otherwise it is reduced over last.
            return tensordot(a, b, {a_ndim - 1}, {std::max(0, b_ndim - 2)});
        } else if (a_ndim == 1 && b_ndim == 1) {
            return vectordot(a, b);
        } else if (a_ndim == 2 && b_ndim == 2) {
            return matrixdot(a, b);
        } else {
            return matrix_vector_dot(a, b);
        }
    }


    Tensor vectordot(const Tensor& a, const Tensor& b) {
        Tensor out(op::vectordot(a.w, b.w));

        if (graph::backprop_enabled())
            graph::emplace_back([a, b, out]() mutable {
                MAYBE_GRAD(a) <<= op::dot(b.w, out.dw);
                MAYBE_GRAD(b) <<= op::dot(a.w, out.dw);
            });
        return out;
    }

    Tensor matrixdot(const Tensor& a, const Tensor& b) {
        Tensor out(op::matrixdot(a.w, b.w));

        if (graph::backprop_enabled())
            graph::emplace_back([a, b, out]() mutable {
                MAYBE_GRAD(a) <<= op::matrixdot(out.dw, b.w.transpose());
                MAYBE_GRAD(b) <<= op::matrixdot(a.w.transpose(), out.dw);
            });
        return out;
    }

    Tensor matrix_vector_dot(const Tensor& a, const Tensor& b) {
        Tensor out(op::matrix_vector_dot(a.w, b.w));

        if (graph::backprop_enabled())
            graph::emplace_back([a, b, out]() mutable {
                if (a.ndim() == 1) {
                    MAYBE_GRAD(a) <<= op::matrix_vector_dot(out.dw, b.w.transpose());
                    MAYBE_GRAD(b) <<= op::outer(a.w, out.dw);
                } else {
                    MAYBE_GRAD(a) <<= op::outer(out.dw, b.w);
                    MAYBE_GRAD(b) <<= op::matrix_vector_dot(a.w.transpose(), out.dw);
                }
            });
        return out;
    }

    Tensor tensordot(const Tensor& a, const Tensor& b, const int& axis) {
        return op::tensordot_as_dot(
            a, b, axis, /*batched=*/false
        );
    }

    Tensor tensordot(const Tensor& a, const Tensor& b,
                     const std::vector<int>& a_reduce_axes,
                     const std::vector<int>& b_reduce_axes) {
         return op::tensordot_as_dot(
             a, b, a_reduce_axes, b_reduce_axes, /*batched=*/false
         );
    }
}  // namespace tensor_ops
