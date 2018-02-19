#include "composite.h"

#include "dali/array/array.h"
#include "dali/array/op/binary.h"
#include "dali/array/op/unary.h"
#include "dali/array/op/dot.h"
#include "dali/tensor/tape.h"
#include "dali/tensor/tensor_macros.h"

namespace tensor_ops {
    Tensor dot_with_bias(const Tensor& input,
                         const Tensor& weight,
                         const Tensor& bias) {
        return multiple_dot_with_bias({input}, {weight}, bias);
    }


    Tensor multiple_dot_with_bias(const std::vector<Tensor>& inputs,
                                  const std::vector<Tensor>& weights,
                                  Tensor bias) {
        ASSERT2(weights.size() == inputs.size(),
                utils::MS() << "Different number of weights and inputs passed "
                            << "to multiple_dot_with_bias ("
                            << "weights.size() = " << weights.size() << " vs. "
                            << "inputs.size() = " << inputs.size() << ").");

        int max_num_examples = 0;
        for (auto input: inputs) {
            max_num_examples = std::max(max_num_examples, input.shape()[0]);
            ASSERT2(input.is_matrix(),
                    utils::MS() << "multiple_dot_with_bias only accepts matrices as inputs"
                                   "(got tensor of shape "  << input.shape() << ")");
        }

        for (auto weight : weights) {
            ASSERT2(weight.is_matrix(),
                    utils::MS() << "multiple_dot_with_bias only accepts matrices as weight"
                                   "(got tensor of shape "  << weight.shape() << ")");
        }

        auto out = Tensor::empty({max_num_examples, weights[0].shape()[1]},
                                 weights[0].dtype(),
                                 weights[0].preferred_device());

        out.w.assign(bias.w);

        for (int i = 0; i < weights.size(); ++i) {
            try {
                out.w += op::dot(inputs[i].w, weights[i].w);
            } catch(std::runtime_error) {
                Array temp = op::dot(inputs[i].w, weights[i].w);
                out.w += temp;
            }
        }
        if (graph::backprop_enabled()) {
            auto out_dw = out.dw;
            auto bias_dw = bias.dw;
            auto bias_constant = bias.constant;
            graph::emplace_back([weights, inputs, bias_dw, bias_constant, out_dw, max_num_examples]() mutable {
                for (int i = 0; i < weights.size(); ++i) {
                    if (inputs[i].w.shape()[0] < 0) {
                        Array temp = op::dot(out_dw, weights[i].w.transpose());
                        MAYBE_GRAD(inputs[i]) <<= temp;
                    } else {
                        MAYBE_GRAD(inputs[i]) <<= op::dot(out_dw, weights[i].w.transpose());
                    }
                    MAYBE_GRAD(weights[i]) <<= op::dot(inputs[i].w.transpose(), out_dw);
                }
                if (!bias_constant)
                    bias_dw <<= out_dw;
            });
        }

        return out;
    }


    Tensor quadratic_form(const Tensor& left, const Tensor& middle, const Tensor& right) {
        if (graph::backprop_enabled()) {
            if (left.is_matrix() && middle.is_matrix() && right.is_matrix()) {
                Array left_side_mul = op::dot(left.w.transpose(), middle.w);
                Tensor out(op::dot(left_side_mul, right.w));
                auto out_dw = out.dw;
                graph::emplace_back([left_side_mul, left, middle, right, out_dw]() mutable {
                    MAYBE_GRAD(right) <<= op::dot(left_side_mul.transpose(), out_dw);
                    Array LeftT_dot_middle_grad = op::dot(out_dw, right.w.transpose());
                    MAYBE_GRAD(left) <<= op::dot(middle.w, LeftT_dot_middle_grad.transpose());
                    MAYBE_GRAD(middle) <<= op::dot(left.w, LeftT_dot_middle_grad);
                });
                return out;
            } else {
                // gradient is different based on tensordot application
                // defer to those gradients computations:
                // TODO(jonathan): make this be memory efficient too through auto-reuse
                return (left.transpose().dot(middle)).dot(right);
            }
        } else {
            return Tensor(op::dot(op::dot(left.w.transpose(), middle.w), right.w));
        }
    }
}  // namespace tensor_ops
