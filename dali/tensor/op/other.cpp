#include "other.h"

#include "dali/tensor/tape.h"
#include "dali/tensor/tensor_macros.h"
#include "dali/array/op.h"

namespace tensor_ops {
    Tensor reshape(const Tensor& t, const std::vector<int>& new_shape) {
        auto out = Tensor::from_w_and_dw(t.w.reshape(new_shape),
                                         t.dw.reshape(new_shape),
                                         t.constant);

        if (t.dw.memory() != out.dw.memory()) {
            // if out.dw is no longer a view, we need backpropagation.
            if (graph::backprop_enabled())
                graph::emplace_back([t, out]() mutable {
                    MAYBE_GRAD(t) <<= out.dw.reshape(t.shape());
                });
        }
        return out;
    }
}  // namespace tensor_ops
