#include "dot.h"
#include "dali/utils/make_message.h"
#include "dali/array/expression/expression.h"

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

    Array tensordot_as_dot(Array a, Array b,
                           const std::vector<int>& a_reduce_axes,
                           const std::vector<int>& b_reduce_axes) {
        throw std::runtime_error("not implemented yet");
    }

    Array dot(Array a, Array b) {
        int a_ndim = a.ndim();
        int b_ndim = b.ndim();
        if (a_ndim == 2 && b_ndim == 2) {
            a = ascontiguousarray_or_simple_transpose(a);
            b = ascontiguousarray_or_simple_transpose(b);
            return Array(std::make_shared<MatMul>(a, b));
        } else if (a_ndim > 2 or b_ndim > 2) {
            return tensordot_as_dot(a, b,
                                    /*a_reduce_axes=*/{a_ndim - 1,},
                                    /*b_reduce_axes=*/{b_ndim - 1,});
        } else {
            throw std::runtime_error(utils::make_message(
                "dot not implemented yet for a.ndim = ", a_ndim,
                ", b.ndim = ", b_ndim));
        }
    }
}
