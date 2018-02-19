#include "binary.h"

#include "dali/array/op/binary.h"
#include "dali/array/op/unary.h"
#include "dali/array/op/circular_convolution.h"
#include "dali/tensor/tape.h"
#include "dali/tensor/tensor_macros.h"

namespace tensor_ops {
    Tensor add(const Tensor& a, const Tensor& b) {
        Tensor out(op::add(a.w, b.w));
        if (graph::backprop_enabled()) {
            auto out_dw = out.dw;
            if (!a.constant) {
                auto a_dw = a.dw;
                graph::emplace_back([a_dw, out_dw]() mutable {
                    a_dw <<= out_dw;
                });
            }
            if (!b.constant) {
                auto b_dw = b.dw;
                graph::emplace_back([b_dw, out_dw]() mutable {
                    b_dw <<= out_dw;
                });
            }
        }
        return out;
    }

    Tensor add(const std::vector<Tensor>& tensors) {
        std::vector<Array> arrays;
        std::vector<Array> grads;
        arrays.reserve(tensors.size());
        for (const auto& t : tensors) {
            arrays.emplace_back(t.w);
            if (!t.constant) {
                grads.emplace_back(t.dw);
            }
        }
        Tensor out(op::add(arrays));
        if (graph::backprop_enabled() && grads.size() > 0) {
            auto out_dw = out.dw;
            graph::emplace_back([grads, out_dw]() mutable {
                for (auto& t_grad : grads) {
                    t_grad <<= out_dw;
                }
            });
        }
        return out;
    }

    Tensor subtract(const Tensor& a, const Tensor& b) {
        Tensor out(a.w - b.w);
        if (graph::backprop_enabled()) {
            auto out_dw = out.dw;
            if (!a.constant) {
                auto a_dw = a.dw;
                graph::emplace_back([a_dw, out_dw]() mutable {
                    a_dw <<= out_dw;
                });
            }
            if (!b.constant) {
                auto b_dw = b.dw;
                graph::emplace_back([b_dw, out_dw]() mutable {
                    b_dw <<= -out_dw;
                });
            }
        }
        return out;
    }

    Tensor eltmul(const Tensor& a, const Tensor& b) {
        Tensor out(op::eltmul(a.w, b.w));

        if (graph::backprop_enabled()) {
            auto out_dw = out.dw;
            graph::emplace_back([a, b, out_dw]() mutable {
                MAYBE_GRAD(a) <<= b.w * out_dw;
                MAYBE_GRAD(b) <<= a.w * out_dw;
            });
        }
        return out;
    }
    Tensor eltdiv(const Tensor& a, const Tensor& b) {
        Tensor out(op::eltdiv(a.w, b.w));
        if (graph::backprop_enabled()) {
            auto out_dw = out.dw;
            graph::emplace_back([a, b, out_dw]() mutable {
                MAYBE_GRAD(a) <<= out_dw / b.w;
                MAYBE_GRAD(b) <<= (-a.w / op::square(b.w)) * out_dw;
            });
        }
        return out;
    }

    Tensor pow(const Tensor& a, const Tensor& exponent) {
        Tensor out(op::pow(a.w, exponent.w));

        if (graph::backprop_enabled())
            graph::emplace_back([a, exponent, out]() mutable {
                MAYBE_GRAD(a) <<=
                        exponent.w * op::pow(a.w, exponent.w - 1.0) * out.dw;
                MAYBE_GRAD(exponent) <<=
                        op::log_or_zero(a.w) * out.w * out.dw;
            });
        return out;
    }

    Tensor circular_convolution(const Tensor& content, const Tensor& shift) {
        Tensor out(op::circular_convolution(content.w, shift.w));
        if (graph::backprop_enabled()) {
            auto out_dw = out.dw;
            graph::emplace_back([content, shift, out_dw]() mutable {
                MAYBE_GRAD(content) <<= op::circular_convolution(out_dw, shift.w);
                MAYBE_GRAD(shift) <<= op::circular_convolution(content.w, out_dw);
            });
        }
        return out;
    }

    Tensor prelu(const Tensor& x, const Tensor& weights) {
        Tensor out(op::prelu(x.w, weights.w));
        if (graph::backprop_enabled()) {
            auto out_dw = out.dw;
            graph::emplace_back([weights, x, out_dw]() mutable {
                MAYBE_GRAD(x) <<= op::prelu_backward_inputs(x.w, weights.w) * out_dw;
                MAYBE_GRAD(weights) <<= op::prelu_backward_weights(x.w, out_dw);
            });
        }
        return out;
    }
}  // namespace tensor_ops
