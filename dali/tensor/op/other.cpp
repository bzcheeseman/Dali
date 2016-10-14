#include "other.h"

#include "dali/array/op2/elementwise_operation.h"
#include "dali/array/op_overload/common.h"
#include "dali/array/op_overload/nonlazy.h"
#include "dali/tensor/tape.h"
#include "dali/tensor/tensor_macros.h"

namespace tensor_ops {
    Tensor reshape(const Tensor& t, const std::vector<int>& new_shape) {
        auto out = Tensor::from_w_and_dw(t.w.reshape(new_shape),
                                         t.dw.reshape(new_shape),
                                         t.constant);

        if (t.dw.memory() != out.dw.memory() && !t.constant && graph::backprop_enabled()) {
            auto out_dw = out.dw;
            auto t_dw = t.dw;
            // if out.dw is no longer a view, we need backpropagation.
            graph::emplace_back([t_dw, out_dw]() mutable {
                t_dw <<= out_dw.reshape(t_dw.shape());
            });
        }
        return out;
    }

    Tensor right_fit_ndim(const Tensor& t, const int& dimensionality) {
        auto out = Tensor::from_w_and_dw(t.w.right_fit_ndim(dimensionality),
                                         t.dw.right_fit_ndim(dimensionality),
                                         t.constant);

        if (t.dw.memory() != out.dw.memory() && !t.constant && graph::backprop_enabled()) {
            auto out_dw = out.dw;
            auto t_dw = t.dw;
            // if out.dw is no longer a view, we need backpropagation.
            graph::emplace_back([t_dw, out_dw]() mutable {
                t_dw <<= out_dw.reshape(t_dw.shape());
            });
        }
        return out;
    }

    Tensor ravel(const Tensor& t) {
        auto out = Tensor::from_w_and_dw(t.w.ravel(),
                                         t.dw.ravel(),
                                         t.constant);

        if (t.dw.memory() != out.dw.memory() && !t.constant && graph::backprop_enabled()) {
            // if out.dw is no longer a view, we need backpropagation.
            auto out_dw = out.dw;
            auto t_dw = t.dw;
            graph::emplace_back([t_dw, out_dw]() mutable {
                t_dw <<= out_dw.reshape(t_dw.shape());
            });
        }
        return out;
    }

    Tensor fill(const Tensor& t, const double& filler) {
        auto out = Tensor::empty_like(t);
        out.w = filler;
        return out;
    }
    Tensor fill(const Tensor& t, const float& filler) {
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

    Tensor consider_constant_if(const Tensor& t, const bool& condition) {
        if (condition) return consider_constant(t);
        return t;
    }

    Tensor consider_constant(const Tensor& t) {
        auto out = t;
        out.constant = true;
        return out;
    }

    bool is_nan(const Tensor& t) {
        return t.w.any_isnan();
    }

    bool is_grad_nan(const Tensor& t) {
        return t.dw.any_isnan();
    }

    bool equals(const Tensor& left, const Tensor& right) {
        return Array::equals(left.w, right.w);
    }

    bool allclose(const Tensor& left, const Tensor& right, const double& atolerance) {
        return Array::allclose(left.w, right.w, atolerance);
    }

    Tensor astype(const Tensor& t, const DType& dtype) {
        if (dtype == t.dtype()) return t;

        Tensor out(op::astype(t.w, dtype));
        out.constant = t.constant || dtype == DTYPE_INT32;

        if (graph::backprop_enabled() && !out.constant) {
            auto out_dw = out.dw;
            auto t_dw = t.dw;
            graph::emplace_back([t_dw, out_dw]() mutable {
                t_dw <<= out_dw;
            });
        }
        return out;
    }


}  // namespace tensor_ops
