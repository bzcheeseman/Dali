#include "cost.h"

#include "dali/array/op/binary.h"
#include "dali/array/op/unary.h"
#include "dali/array/op/gather_from_rows.h"
#include "dali/array/op/reducers.h"
#include "dali/array/op/softmax.h"
#include "dali/tensor/tape.h"
#include "dali/utils/make_message.h"
#include "dali/tensor/tensor_macros.h"

namespace tensor_ops {
    Tensor binary_cross_entropy(const Tensor& t, const double& target) {
        ASSERT2(0 <= target && target <= 1,
                utils::MS() << "Target value for binary_cross_entropy must be a "
                            << "real number between 0 and 1 (got " << target << ").");

        Tensor out(op::binary_cross_entropy(t.w, target));
        if (graph::backprop_enabled()) {
            graph::emplace_back([t, target, out]() mutable {
                MAYBE_GRAD(t) <<= op::binary_cross_entropy_grad(t.w, target) * out.dw;
            });
        }
        return out;
    }

    Tensor binary_cross_entropy(const Tensor& t, const Tensor& target) {
        Tensor out(op::binary_cross_entropy(t.w, target.w));
        if (graph::backprop_enabled()) {
            graph::emplace_back([t, target, out]() mutable {
                MAYBE_GRAD(t) <<= op::binary_cross_entropy_grad(t.w, target.w) * out.dw;
                MAYBE_GRAD(target) <<= 2.0 * op::inverse_tanh(1.0 - 2.0 * t.w) * out.dw;
            });
        }
        return out;
    }

    Tensor sigmoid_binary_cross_entropy(const Tensor& t, const double& target) {
        ASSERT2(0 <= target && target <= 1,
                utils::MS() << "Target value for binary_cross_entropy must be a "
                            << "real number between 0 and 1 (got " << target << ").");
        Tensor out(op::binary_cross_entropy(op::sigmoid(t.w), target));
        if (graph::backprop_enabled()) {
            graph::emplace_back([t, target, out]() mutable {
                MAYBE_GRAD(t) <<= (op::sigmoid(t.w) - target) * out.dw;
            });
        }
        return out;
    }

    Tensor sigmoid_binary_cross_entropy(const Tensor& t, const Tensor& target) {
        Tensor out(op::binary_cross_entropy(op::sigmoid(t.w), target.w));
        if (graph::backprop_enabled()) {
            graph::emplace_back([t, target, out]() mutable {
                MAYBE_GRAD(t) <<= (op::sigmoid(t.w) - target.w) * out.dw;
                MAYBE_GRAD(target) <<= 2.0 * op::inverse_tanh(1.0 - 2.0 * op::sigmoid(t.w)) * out.dw;
            });
        }
        return out;
    }

    Tensor margin_loss(const Tensor& t, const int& target, const double& margin, const int& axis) {
        // relevant slice:
        auto t_plucked = t.pluck_axis(axis, target);
        t_plucked = t_plucked.insert_broadcast_axis(axis);
        // now compare slice to rest of elements:
        Tensor out(op::eltmax(t_plucked.w - t.w + margin, 0.0));
        out.pluck_axis(axis, target).w = 0;

        // gradient is wherever the
        if (graph::backprop_enabled()) {
            graph::emplace_back([t, t_plucked, margin, out, axis, target]() mutable {

                // TODO(szymon, jonathan): implement gradient -- previous gradient was not done explicitly

                // MAYBE_GRAD(t) <<= -op::F<functor::greaterthanequal>(out.w, 0.0) * out.dw;

                // auto t_at_target = t.pluck_axis(axis, target);
                // MAYBE_GRAD(t_at_target) <<= (
                //     op::F<functor::greaterthanequal>(out.pluck_axis(axis, target).w, 0.0) *
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
        ASSERT2(false, "Margin loss is not yet implemented.");
        return Tensor();
    }

    Tensor softmax(const Tensor& t, int axis, const double& temperature) {
        if (axis < 0) axis = t.ndim() + axis;

        Tensor out(op::softmax(t.w, axis, temperature));

        if (graph::backprop_enabled() && !t.constant)
            graph::emplace_back([t, out, axis, temperature]() {
                Array softmax_times_grad_summed(op::sum(out.w * out.dw, {axis}));
                auto expr = (out.w * out.dw) - (out.w * softmax_times_grad_summed.insert_broadcast_axis(axis));
                if (std::abs(temperature - 1.0) < 1e-6) {
                    MAYBE_GRAD(t) <<= expr;
                } else {
                    MAYBE_GRAD(t) <<= expr / temperature;
                }
            });
        return out;
    }

    Tensor softmax_cross_entropy_with_idxes(const Tensor& unnormalized_probs,
                                            const Tensor& targets,
                                            const double& temperature,
                                            const int& axis) {
        ASSERT2(axis == unnormalized_probs.ndim() -1, utils::make_message(
            "softmax_cross_entropy is not yet implemented for axes other than "
            "-1 (got axis=", axis, ")"));
        Array softmax_result = op::softmax(unnormalized_probs.w, axis, temperature);
        Tensor out(-1.0 * op::log(op::gather_from_rows(softmax_result, targets.w)));

        if (graph::backprop_enabled() && !unnormalized_probs.constant)
            graph::emplace_back([unnormalized_probs, softmax_result, targets, temperature, axis, out]() {
                unnormalized_probs.dw <<= softmax_result * out.dw.insert_broadcast_axis(axis) / temperature;
                unnormalized_probs.dw.gather_from_rows(targets.w) -= out.dw / temperature;
            });

        return out;
    }


    Tensor softmax_cross_entropy_with_probs(const Tensor& unnormalized_probs,
                                            const Tensor& targets,
                                            const double& temperature,
                                            const int& axis) {
        Array softmax_result = op::softmax(unnormalized_probs.w, axis, temperature);
        Tensor out(-1.0 * targets.w * op::log(softmax_result));

        if (graph::backprop_enabled())
            graph::emplace_back([unnormalized_probs, softmax_result, targets, temperature, axis, out]() {
                if (!unnormalized_probs.constant) {

                    Array grad_times_target = op::sum(targets.w * out.dw, {axis});
                    grad_times_target = grad_times_target.insert_broadcast_axis(axis);

                    unnormalized_probs.dw <<=
                        softmax_result * grad_times_target / temperature
                        - targets.w * out.dw / temperature;
                }
                MAYBE_GRAD(targets) <<= -op::log(softmax_result) * out.dw;
            });
        return out;
    }


    Tensor softmax_cross_entropy(const Tensor& unnormalized_probs,
                                 const Tensor& targets,
                                 const double& temperature,
                                 int axis) {

        if (axis < 0) axis = unnormalized_probs.ndim() + axis;

        if (targets.dtype() == DTYPE_INT32) {
            return softmax_cross_entropy_with_idxes(unnormalized_probs, targets, temperature, axis);
        } else {
            return softmax_cross_entropy_with_probs(unnormalized_probs, targets, temperature, axis);
        }
    }

    Tensor cross_entropy_with_idxes(const Tensor& probs, const Tensor& target, int axis) {
        // 1) we ensure that cross entropy is happening in a 1D setting or
        // else the behavior of gather_from_rows would not broadcast in the
        // desired way (e.g. we assume that a single dimension contains a
        // multinomial distribution for which KL divergence must be computed).
        if (target.ndim() > 1) {
            return cross_entropy_with_idxes(
                probs.swapaxes(-1, axis).right_fit_ndim(2),
                target.ravel(),
                -1
            ).reshape(target.shape()).swapaxes(-1, axis);
        }
        // 2) We now know that input probs are of the right dimensionality for
        // our problem (e.g. they should be 2D) and that our targets are 1D:
        auto permuted_probs = probs.swapaxes(-1, axis);
        Tensor out(-1.0 * op::log(op::gather_from_rows(permuted_probs.w, target.w)));

        if (graph::backprop_enabled())
            graph::emplace_back([permuted_probs, target, out]() {
                if (!permuted_probs.constant) {
                    permuted_probs.dw.gather_from_rows(target.w) +=
                        -out.dw / op::gather_from_rows(permuted_probs.w, target.w);
                }
            });
        return out.swapaxes(axis, -1);
    }

    Tensor cross_entropy_with_probs(const Tensor& probs, const Tensor& target) {
        Tensor out(-1.0 * target.w * op::log(probs.w));

        if (graph::backprop_enabled())
            graph::emplace_back([probs, target, out]() {
                MAYBE_GRAD(probs) <<= -op::eltinv(probs.w) * target.w * out.dw;
                MAYBE_GRAD(target) <<= -op::log(probs.w) * out.dw;
            });
        return out;
    }

    Tensor cross_entropy(const Tensor& probs, const Tensor& target, int axis) {
        if (target.dtype() == DTYPE_INT32) {
            return cross_entropy_with_idxes(probs, target, axis);
        } else {
            return cross_entropy_with_probs(probs, target);
        }
    }


}  // namespace tensor_ops
