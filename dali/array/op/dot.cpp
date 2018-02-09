#include "dot.h"
#include "dali/utils/make_message.h"
#include "dali/array/expression/expression.h"
#include "dali/array/gemm/tensordot_as_dot.h"
#include "dali/array/jit/reshape.h"

// DOT SPECIFIC CLASSES
namespace {
    Array ascontiguousarray_or_simple_transpose(Array node) {
        auto buff = node.buffer_arg();
        if (!buff.is_stateless() && (buff.contiguous_memory() or buff.is_transpose())) {
            return node;
        }
        return node.ascontiguousarray();
    }
}

namespace op {
    Array matrix_multiply_with_reshape(const Array& a,
                                       const Array& b,
                                       const std::vector<int>& out_shape,
                                       const std::vector<int>& out_shape_2d) {
        return matmul(a, b).reshape(out_shape);
    }

    MatMul::MatMul(Array left, Array right) :
        Expression({left.shape()[0], right.shape()[1]},
                   type_promotion(left, right), {left, right}) {}
    expression_ptr MatMul::copy() const {
        return std::make_shared<MatMul>(*this);
    }
    memory::Device MatMul::preferred_device() const {
        return device_promotion(arguments_[0], arguments_[1]);
    }

    bool MatMul::supports_operator(OPERATOR_T operator_t) const {
        return (operator_t == OPERATOR_T_EQL ||
                operator_t == OPERATOR_T_ADD ||
                operator_t == OPERATOR_T_SUB);
    }

    Array matmul(Array a, Array b) {
        ASSERT2(a.ndim() == 2 && b.ndim() == 2, utils::make_message(
            "matmul inputs must be both be 2-dimensional, but got a.ndim = ",
            a.ndim(), " and b.ndim = ", b.ndim(), "."));
        ASSERT2(a.shape()[1] == b.shape()[0] | a.shape()[1] == 1 | b.shape()[0] == 1,
            utils::make_message("matmul shapes not aligned, with a.shape = ", a.shape(),
                " and b.shape = ", b.shape(), ": expected a.shape[1] = ", a.shape()[1],
                " to equal b.shape[0] = ", b.shape()[0], ", or for either to equal 1."));
        if (a.shape()[1] != b.shape()[0]) {
            if (a.shape()[1] == 1) {
                a = op::jit::broadcasted_reshape(a, {a.shape()[0], b.shape()[0]});
            } else {
                b = op::jit::broadcasted_reshape(b, {a.shape()[1], b.shape()[1]});
            }
        }
        a = ascontiguousarray_or_simple_transpose(a);
        b = ascontiguousarray_or_simple_transpose(b);
        return Array(std::make_shared<MatMul>(a, b));
    }

    Array matrix_vector_dot(const Array& a, const Array& b) {
        // TODO(jonathan): use correct blas subroutine whenever possible (gemv)
        ASSERT2((a.ndim() == 1 && b.ndim() == 2) || (a.ndim() == 2 && b.ndim() == 1),
            utils::make_message("matrix_vector_dot inputs must be a vector and a matrix,"
                " but got a.ndim = ", a.ndim(), " and b.ndim = ", b.ndim(), "."));
        if (a.ndim() == 1 && b.ndim() == 2) {
            ASSERT2(b.shape()[0] == a.shape()[0], utils::make_message(
                "matrix_vector_dot shape mistmach between a.shape = ", a.shape(),
                " and b.shape = ", b.shape(), ": a.shape[0] = ", a.shape()[0],
                " != b.shape[0] = ", b.shape()[0], "."));
            return matmul(a.reshape({1, a.number_of_elements()}),
                          b).reshape({b.shape()[1]});
        } else {
            ASSERT2(a.shape()[1] == b.shape()[0], utils::make_message(
                "matrix_vector_dot shape mistmach between a.shape = ", a.shape(),
                " and b.shape = ", b.shape(), ": a.shape[1] = ", a.shape()[1],
                " != b.shape[0] = ", b.shape()[0], "."));
            return matmul(a, b.reshape({b.number_of_elements(), 1})).reshape(
                {a.shape()[0]});
        }
    }

    Array tensordot(const Array& a, const Array& b, int axis) {
        return tensordot_as_dot(a, b, axis, /*batched=*/false);
    }

    Array tensordot(const Array& a, const Array& b, const std::vector<int>& a_reduce_axes, const std::vector<int>& b_reduce_axes) {
        return tensordot_as_dot(a, b, a_reduce_axes, b_reduce_axes, /*batched=*/false);
    }

    Array inner(const Array& a, const Array& b) {
        ASSERT2(a.ndim() == 1 && b.ndim() == 1, utils::make_message("inner expects vector inputs"
            " but got a.ndim = ", a.ndim(), " and b.ndim = ", b.ndim(), " arrays."));
        ASSERT2(a.shape()[0] == b.shape()[0], utils::make_message("inner input shapes a.shape = ",
            a.shape(), " and b.shape = ", b.shape(), " should be equal."));
        return matmul(a.reshape({1, a.number_of_elements()}),
                      b.reshape({b.number_of_elements(), 1})).reshape({});
    }

    Array dot(const Array& a, const Array& b) {
        int a_ndim = a.ndim();
        int b_ndim = b.ndim();
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
            return tensordot(a, b,
                             /*a_reduce_axes=*/{a_ndim - 1},
                             /*b_reduce_axes=*/{std::max(0, b_ndim - 2)});
        } else if (a_ndim == 1 && b_ndim == 1) {
            return inner(a, b);
        } if (a_ndim == 2 && b_ndim == 2) {
            return matmul(a, b);
        } else {
            return matrix_vector_dot(a, b);
        }
    }
}
