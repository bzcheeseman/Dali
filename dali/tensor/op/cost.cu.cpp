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


}  // namespace tensor_ops
