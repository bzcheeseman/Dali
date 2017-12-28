#include "dropout.h"

#include "dali/tensor/tape.h"
#include "dali/tensor/tensor_macros.h"
#include "dali/array/op/initializer.h"
#include "dali/array/op.h"


namespace tensor_ops {

    Tensor dropout_unnormalized(const Tensor& t, const double& drop_prob) {
        ASSERT2(0.0 <= drop_prob && drop_prob <= 1.0, utils::make_message(
            "drop_prob argument to dropout_unnormalized must be in the interval 0.0 to 1.0 (got ",
            drop_prob, ")."));
        // Skip noise if probability is too low:
        if (drop_prob < 1e-6) return t;

        auto mask = Array::empty_like(t.w);
        mask = initializer::bernoulli(1.0 - drop_prob);
        Tensor out(t.w * mask);

        if (graph::backprop_enabled() && !t.constant) {
            auto t_dw = t.dw;
            auto out_dw = out.dw;
            graph::emplace_back([t_dw, out_dw, mask]() mutable {
                t_dw <<= out_dw * mask;
            });
        }
        return out;
    }

    Tensor dropout(const Tensor& t, const double& drop_prob) {
        ASSERT2(0.0 <= drop_prob && drop_prob <= 1.0, utils::make_message(
            "drop_prob argument to dropout must be in the interval 0.0 to 1.0 (got ",
            drop_prob, ")."));
        // Skip noise if probability is too low:
        if (drop_prob < 1e-6) return t;

        auto mask = Array::empty_like(t.w);
        mask = initializer::bernoulli_normalized(1.0 - drop_prob);
        Tensor out(t.w * mask);

        if (graph::backprop_enabled() && !t.constant) {
            auto t_dw = t.dw;
            auto out_dw = out.dw;
            graph::emplace_back([t_dw, out_dw, mask]() mutable {
                t_dw <<= out_dw * mask;
            });
        }
        return out;
    }

    Tensor fast_dropout(const Tensor& t) {
        auto mask = Array::empty_like(t.w);
        mask = initializer::gaussian(1.0, 1.0);
        Tensor out(t.w * mask);

        if (graph::backprop_enabled() && !t.constant) {
            auto t_dw = t.dw;
            auto out_dw = out.dw;
            graph::emplace_back([t_dw, out_dw, mask]() mutable {
                t_dw <<= out_dw * mask;
            });
        }
        return out;
    }
}
