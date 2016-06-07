#include "cost.h"

#include "dali/array/functor.h"
#include "dali/array/lazy_op.h"
#include "dali/tensor/tape.h"
#include "dali/tensor/tensor_macros.h"
#include "dali/array/lazy/binary.h"


namespace tensor_ops {
    Tensor binary_cross_entropy(const Tensor& t, const double& target) {
        ASSERT2(0 <= target && target <= 1,
                utils::MS() << "Target value for binary_cross_entropy must be a "
                            << "real number between 0 and 1 (got " << target << ").");

        Tensor out(lazy::F<functor::binary_cross_entropy>(t.w, target));
        if (graph::backprop_enabled()) {
            graph::emplace_back([t, target, out]() mutable {
                MAYBE_GRAD(t) <<= lazy::F<functor::binary_cross_entropy_grad>(t.w, target) * out.dw;
            });
        }
        return out;
    }

    Tensor binary_cross_entropy(const Tensor& t, const Tensor& target) {
        Tensor out(lazy::F<functor::binary_cross_entropy>(t.w, target.w));
        if (graph::backprop_enabled()) {
            graph::emplace_back([t, target, out]() mutable {
                MAYBE_GRAD(t) <<= lazy::F<functor::binary_cross_entropy_grad>(t.w, target.w) * out.dw;
                MAYBE_GRAD(target) <<= 2.0 * lazy::F<functor::inverse_tanh>(1.0 - 2.0 * t.w) * out.dw;
            });
        }
        return out;
    }

    Tensor sigmoid_binary_cross_entropy(const Tensor& t, const double& target) {
        ASSERT2(0 <= target && target <= 1,
                utils::MS() << "Target value for binary_cross_entropy must be a "
                            << "real number between 0 and 1 (got " << target << ").");
        Tensor out(lazy::F<functor::binary_cross_entropy>(lazy::sigmoid(t.w), target));
        if (graph::backprop_enabled()) {
            graph::emplace_back([t, target, out]() mutable {
                MAYBE_GRAD(t) <<= (lazy::sigmoid(t.w) - target) * out.dw;
            });
        }
        return out;
    }

    Tensor sigmoid_binary_cross_entropy(const Tensor& t, const Tensor& target) {
        Tensor out(lazy::F<functor::binary_cross_entropy>(lazy::sigmoid(t.w), target.w));
        if (graph::backprop_enabled()) {
            graph::emplace_back([t, target, out]() mutable {
                MAYBE_GRAD(t) <<= (lazy::sigmoid(t.w) - target.w) * out.dw;
                MAYBE_GRAD(target) <<= 2.0 * lazy::F<functor::inverse_tanh>(1.0 - 2.0 * lazy::sigmoid(t.w)) * out.dw;
            });
        }
        return out;
    }

    Tensor margin_loss(const Tensor& t, const int& target, const double& margin, const int& axis) {
        // relevant slice:
        auto t_plucked = t.pluck_axis(axis, target);
        t_plucked = t_plucked.insert_broadcast_axis(axis);
        // now compare slice to rest of elements:
        Tensor out(lazy::eltmax(t_plucked.w - t.w + margin, 0.0));
        out.pluck_axis(axis, target).w = 0;

        // gradient is wherever the
        if (graph::backprop_enabled()) {
            graph::emplace_back([t, t_plucked, margin, out, axis, target]() mutable {

                // TODO(szymon, jonathan): implement gradient -- previous gradient was not done explicitly

                // MAYBE_GRAD(t) <<= -lazy::F<functor::greaterthanequal>(out.w, 0.0) * out.dw;

                // auto t_at_target = t.pluck_axis(axis, target);
                // MAYBE_GRAD(t_at_target) <<= (
                //     lazy::F<functor::greaterthanequal>(out.pluck_axis(axis, target).w, 0.0) *
                //     out.pluck_axis(axis, target).dw
                // );

                // auto out_subslice = out.pluck_axis(axis, target);
                // out_subslice = out_subslice.insert_broadcast_axis(axis);
                // MAYBE_GRAD(t_plucked) <<= out_subslice.dw;
            });
        }
        return out;
    }

    Tensor margin_loss(const Tensor&, const Tensor& target, const double& margin, const int& axis) {
        ASSERT2(false, "not implemented");
        return Tensor();
    }

    Tensor softmax(const Tensor& t, int axis, const double& temperature) {
        if (axis < 0) axis = t.ndim() + axis;

        Tensor out(op::softmax(t.w, axis, temperature));

        if (graph::backprop_enabled() && !t.constant)
            graph::emplace_back([t, out, axis, temperature]() {
                Array softmax_times_grad_summed(lazy::sum(out.w * out.dw, axis));
                auto expr = (out.w * out.dw) - (out.w * softmax_times_grad_summed.insert_broadcast_axis(axis));
                if (std::abs(temperature - 1.0) < 1e-6) {
                    MAYBE_GRAD(t) <<= expr;
                } else {
                    MAYBE_GRAD(t) <<= expr / temperature;
                }
            });
        return out;
    }

    // Tensor softmax_cross_entropy_rowwise(const Tensor& t, const Tensor& targets, int axis) {
    //     if (axis < 0) axis = t.ndim() + axis;
    //     Array probs(op::softmax(t.w, axis, 1.0));

    //     Tensor out(-1.0 * lazy::negative_log(lazy::take(probs, targets.w)));

    //     if (graph::backprop_enabled() && !t.constant) {
    //         graph::emplace_back([t, probs, out, targets, axis]() mutable {
    //             MAYBE_GRAD(t) <<= probs * out.insert_broadcast_axis(axis).w

    //             if (!matrix.constant) {
    //                 GRAD(matrix) += (
    //                     MAT(probs).wrapper() *
    //                     GRAD(out).ravel().wrapper().template broadcast<0>(MAT(probs).shape)
    //                 );

    //                 softmax_cross_entropy_rowwise_backward(GRAD(matrix), GRAD(out), targets.w().ravel());
    //             }
    //         });
    //     }
    //     return out;
    // }

    Tensor cross_entropy(const Tensor& probs, const Tensor& target) {
        Tensor out(-1.0 * target.w * lazy::log(probs.w));

        if (graph::backprop_enabled())
            graph::emplace_back([probs, target, out]() {
                MAYBE_GRAD(probs) <<= -lazy::eltinv(probs.w) * target.w * out.dw;
                MAYBE_GRAD(target) <<= -lazy::log(probs.w) * out.dw;
            });
        return out;
    }

}  // namespace tensor_ops
