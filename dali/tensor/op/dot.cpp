#include "dot.h"

#include "dali/array/gemm/tensordot_as_dot.h"
#include "dali/tensor/tape.h"
#include "dali/tensor/tensor_macros.h"
#include "dali/array/op.h"

Tensor matmul_with_custom_shape(const Tensor& a,
                                   const Tensor& b,
                                   std::vector<int> new_shape) {
    auto out = Tensor::empty(new_shape, a.dtype(), a.preferred_device());
    out.w = op::matmul(a.w, b.w);

    if (graph::backprop_enabled()) {
        auto out_dw = out.dw;
        graph::emplace_back([a, b, out_dw]() mutable {
            MAYBE_GRAD(a) <<= op::matmul(out_dw, b.w.transpose());
            MAYBE_GRAD(b) <<= op::matmul(a.w.transpose(), out_dw);
        });
    }
    return out;
}

namespace op {
    template<>
    Tensor matrix_multiply_with_reshape(const Tensor& a,
                                        const Tensor& b,
                                        const std::vector<int>& out_shape,
                                        const std::vector<int>& out_shape_2d) {
        return matmul_with_custom_shape(a, b, out_shape_2d).reshape(out_shape);
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
            return inner(a, b);
        } else if (a_ndim == 2 && b_ndim == 2) {
            return matmul(a, b);
        } else {
            return matrix_vector_dot(a, b);
        }
    }

    Tensor outer(const Tensor& a, const Tensor& b) {
        Tensor out(op::outer(a.w, b.w));
        if (graph::backprop_enabled()) {
            auto out_dw = out.dw;
            graph::emplace_back([a, b, out_dw]() mutable {
                MAYBE_GRAD(a) <<= op::dot(out_dw, b.w);
                MAYBE_GRAD(b) <<= op::dot(a.w, out_dw);
            });
        }
        return out;
    }

    Tensor inner(const Tensor& a, const Tensor& b) {
        Tensor out(op::inner(a.w, b.w));

        if (graph::backprop_enabled()) {
            auto out_dw = out.dw;
            graph::emplace_back([a, b, out_dw]() mutable {
                MAYBE_GRAD(a) <<= op::dot(b.w, out_dw);
                MAYBE_GRAD(b) <<= op::dot(a.w, out_dw);
            });
        }
        return out;
    }

    Tensor matmul(const Tensor& a, const Tensor& b) {
        Tensor out(op::matmul(a.w, b.w));

        if (graph::backprop_enabled()) {
            auto out_dw = out.dw;
            graph::emplace_back([a, b, out_dw]() mutable {
                MAYBE_GRAD(a) <<= op::matmul(out_dw, b.w.transpose());
                MAYBE_GRAD(b) <<= op::matmul(a.w.transpose(), out_dw);
            });
        }
        return out;
    }

    Tensor matrix_vector_dot(const Tensor& a, const Tensor& b) {
        Tensor out(op::matrix_vector_dot(a.w, b.w));

        if (graph::backprop_enabled()) {
            auto out_dw = out.dw;
            graph::emplace_back([a, b, out_dw]() mutable {
                if (a.ndim() == 1) {
                    MAYBE_GRAD(a) <<= op::matrix_vector_dot(out_dw, b.w.transpose());
                    MAYBE_GRAD(b) <<= op::outer(a.w, out_dw);
                } else {
                    MAYBE_GRAD(a) <<= op::outer(out_dw, b.w);
                    MAYBE_GRAD(b) <<= op::matrix_vector_dot(a.w.transpose(), out_dw);
                }
            });
        }
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
