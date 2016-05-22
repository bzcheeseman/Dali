#include "other.h"

#include "dali/tensor/tape.h"
#include "dali/tensor/tensor_macros.h"

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

    Tensor fill(const Tensor& t, const double& filler) {
        auto out = Tensor::empty_like(t);
        out.w = filler;
        return out;
    }
    Tensor fill(const Tensor& t, const float& filler) {
        auto out = Tensor::empty_like(t);
        out.w = filler;
        return out;
    }
    Tensor fill(const Tensor& t, const int& filler) {
        auto out = Tensor::empty_like(t);
        out.w = filler;
        return out;
    }

    void grad(const Tensor& t) {
        t.grad();
    }

    Tensor consider_constant_if(const Tensor& t, const bool& condition) {
        if (condition) return consider_constant(t);
        return t;
    }

    Tensor consider_constant(const Tensor& t) {
        auto out = t;
        out.constant = true;
        return out;
    }

}  // namespace tensor_ops
