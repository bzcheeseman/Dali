#include "binary.h"

#include "dali/array/lazy_op.h"
#include "dali/tensor/tape.h"
#include "dali/tensor/tensor_macros.h"

namespace tensor_ops {
    Tensor add(const Tensor& a, const Tensor& b) {
        Tensor out(op::add(a.w, b.w));

        if (graph::backprop_enabled())
            graph::emplace_back([a, b, out]() mutable {
                MAYBE_GRAD(a) <<= out.dw;
                MAYBE_GRAD(b) <<= out.dw;
            });
        return out;
    }

    Tensor add(const std::vector<Tensor>& tensors) {
        std::vector<Array> arrays;
        arrays.reserve(tensors.size());
        for (const auto& t : tensors) {
            arrays.emplace_back(t.w);
        }
        Tensor out(op::add(arrays));

        if (graph::backprop_enabled())
            graph::emplace_back([tensors, out]() {
                for (const auto& t : tensors) {
                    MAYBE_GRAD(t) <<= out.dw;
                }
            });
        return out;
    }

    Tensor sub(const Tensor& a, const Tensor& b) {
        Tensor out(op::sub(a.w, b.w));

        if (graph::backprop_enabled())
            graph::emplace_back([a, b, out]() mutable {
                MAYBE_GRAD(a) <<= out.dw;
                MAYBE_GRAD(b) <<= -out.dw;
            });
        return out;
    }

    Tensor eltmul(const Tensor& a, const Tensor& b) {
        Tensor out(op::eltmul(a.w, b.w));

        if (graph::backprop_enabled())
            graph::emplace_back([a, b, out]() mutable {
                MAYBE_GRAD(a) <<= b.w * out.dw;
                MAYBE_GRAD(b) <<= a.w * out.dw;
            });
        return out;
    }
    Tensor eltdiv(const Tensor& a, const Tensor& b) {
        Tensor out(op::eltdiv(a.w, b.w));

        if (graph::backprop_enabled())
            graph::emplace_back([a, b, out]() mutable {
                MAYBE_GRAD(a) <<= out.dw / b.w;
                MAYBE_GRAD(b) <<= (-a.w / lazy::square(b.w)) * out.dw;
            });
        return out;
    }

    Tensor pow(const Tensor& a, const Tensor& exponent) {
        Tensor out(lazy::pow(a.w, exponent.w));

        if (graph::backprop_enabled())
            graph::emplace_back([a, exponent, out]() mutable {
                MAYBE_GRAD(a) <<=
                        exponent.w * lazy::pow(a.w, exponent.w - 1.0) * out.dw;
                MAYBE_GRAD(exponent) <<=
                        lazy::log_or_zero(a.w) * out.w * out.dw;
            });
        return out;
    }
}  // namespace tensor_ops
