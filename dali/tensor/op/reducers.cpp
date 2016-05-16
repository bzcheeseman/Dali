#include "reducers.h"
#include "dali/tensor/tape.h"
#include "dali/array/op.h"
#include "dali/utils/print_utils.h"

namespace tensor_ops {
    Tensor sum(const Tensor& tensor) {
        if (tensor.number_of_elements() == 1) {
            auto out = tensor;
            out.w = tensor.w.reshape({});
            out.dw = tensor.dw.reshape({});
            return out;
        } else {
            // TODO(jonathan, szymon) also makes sure that device
            // of input tensor is also used here

            auto out = Tensor(tensor.w.sum());

            if (graph::backprop_enabled() && !tensor.constant)
                graph::emplace_back([tensor, out]() mutable {
                    tensor.dw <<= out.dw.broadcast_scalar_to_ndim(tensor.ndim());
                });
            return out;
        }
    }

    Tensor mean(const Tensor& tensor) {
        if (tensor.number_of_elements() == 1) {
            auto out = tensor;
            out.w = tensor.w.reshape({});
            out.dw = tensor.dw.reshape({});
            return out;
        } else {
            Tensor out(tensor.w.mean());
            if (graph::backprop_enabled() && !tensor.constant)
                graph::emplace_back([tensor, out]() mutable {
                    tensor.dw <<= (
                        out.dw.broadcast_scalar_to_ndim(tensor.ndim()) /
                        tensor.number_of_elements()
                    );
                });
            return out;
        }
    }

    Tensor sum(const Tensor& tensor, const int& axis) {
        Tensor out(op::sum(tensor.w, axis));
        if (graph::backprop_enabled() && !tensor.constant)
            graph::emplace_back([tensor, out, axis]() mutable {
                // make sure output has same shape as input
                // with the reduced dimension returned as
                // broadcasted
                auto reshaped_gradient = out.dw.insert_broadcast_axis(axis);
                tensor.dw <<= reshaped_gradient;
            });
        return out;
    }

    Tensor mean(const Tensor& tensor, const int& axis) {
        Tensor out(op::mean(tensor.w, axis));
        if (graph::backprop_enabled() && !tensor.constant)
            graph::emplace_back([tensor, out, axis]() mutable {
                int axis_size = tensor.shape()[axis];
                auto reshaped_gradient = out.dw.insert_broadcast_axis(axis);
                tensor.dw <<= reshaped_gradient / axis_size;
            });
        return out;
    }
}
