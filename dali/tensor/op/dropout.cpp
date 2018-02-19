#include "dropout.h"

#include "dali/tensor/tape.h"
#include "dali/tensor/tensor_macros.h"
#include "dali/array/op.h"
#include "dali/utils/make_message.h"


namespace tensor_ops {

    Tensor dropout_unnormalized(const Tensor& t, const double& drop_prob) {
        ASSERT2(0.0 <= drop_prob && drop_prob <= 1.0, utils::make_message(
            "dropout_unnormalized's drop_prob argument must be in the interval 0.0 to 1.0 (got ",
            drop_prob, ")."));
        // Skip noise if probability is too low:
        if (drop_prob < 1e-6) return t;
        auto mask = op::bernoulli(Array(1.0 - drop_prob, t.dtype()), t.w.shape());
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
        auto mask = op::bernoulli_normalized(Array(1.0 - drop_prob, t.dtype()), t.w.shape());
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
        auto mask = op::normal(Array(1.0, t.dtype()), Array(1.0, t.dtype()), t.shape());
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
