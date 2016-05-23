#include "cost.h"

#include "dali/array/functor.h"
#include "dali/array/lazy_op.h"
#include "dali/tensor/tape.h"
#include "dali/tensor/tensor_macros.h"
#include "dali/array/lazy/binary.h"


namespace tensor_ops {
    Tensor binary_cross_entropy(const Tensor& t, const double& target) {
        ASSERT2(0 <= target && target <= 1,
                "Target value for binary_cross_entropy must be a probability between 0 and 1.");

        Tensor out(lazy::F<functor::binary_cross_entropy>(t.w, target));
        if (graph::backprop_enabled()) {
            graph::emplace_back([t, target, out]() mutable {
                MAYBE_GRAD(t) <<= lazy::F<functor::binary_cross_entropy_grad>(t.w, target) * out.dw;
            });
        }
        return out;
    }

    Tensor binary_cross_entropy(const Tensor& t, const Tensor& target) {
        ASSERT2(t.shape() == target.shape(),
                "binary_cross_entropy input and target must have the same shape.");

        Tensor out(lazy::F<functor::binary_cross_entropy>(t.w, target.w));
        if (graph::backprop_enabled()) {
            graph::emplace_back([t, target, out]() mutable {
                MAYBE_GRAD(t) <<= lazy::F<functor::binary_cross_entropy_grad>(t.w, target.w) * out.dw;
                // TODO(crazy_people): add grad through target.
            });
        }
        return out;
    }
}  // namespace tensor_ops
