#include "binary.h"

#include "dali/array/lazy_op.h"
#include "dali/tensor/tape.h"
#include "dali/tensor/tensor_macros.h"


namespace tensor_ops {
    Tensor add(Tensor a, Tensor b) {
        Tensor out(a.w + b.w);

        if (graph::backprop_enabled())
            graph::emplace_back([a, b, out]() mutable {
                MAYBE_GRAD(a) <<= out.dw;
                MAYBE_GRAD(b) <<= out.dw;
            });
        return out;
    }

    Tensor sub(Tensor a, Tensor b) {
        Tensor out(a.w - b.w);

        if (graph::backprop_enabled())
            graph::emplace_back([a, b, out]() mutable {
                MAYBE_GRAD(a) <<= out.dw;
                MAYBE_GRAD(b) <<= -out.dw;
            });
        return out;
    }

    Tensor eltmul(Tensor a, Tensor b) {
        Tensor out(a.w * b.w);

        if (graph::backprop_enabled())
            graph::emplace_back([a, b, out]() mutable {
                MAYBE_GRAD(a) <<= b.w * out.dw;
                MAYBE_GRAD(b) <<= a.w * out.dw;
            });
        return out;
    }
    Tensor eltdiv(Tensor a, Tensor b) {
        Tensor out(a.w / b.w);

        if (graph::backprop_enabled())
            graph::emplace_back([a, b, out]() mutable {
                MAYBE_GRAD(a) <<= out.dw / b.w;
                MAYBE_GRAD(b) <<= (-a.w / lazy::square(b.w)) * out.dw;
            });
        return out;
    }

    Tensor pow(Tensor a, Tensor exponent) {
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

    Tensor dot(Tensor a, Tensor b) {
        Tensor out(op::dot(a.w, b.w));

        if (graph::backprop_enabled())
            graph::emplace_back([a, b, out]() mutable {
                MAYBE_GRAD(a) <<= op::dot(out.dw, b.w.transpose());
                MAYBE_GRAD(b) <<= op::dot(a.w.transpose(), out.dw);
            });
        return out;
    }
}
