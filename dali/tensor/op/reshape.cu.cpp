#include "reshape.h"
#include "dali/tensor/tape.h"
#include "dali/array/op/reshape.h"
#include "dali/tensor/tensor_macros.h"

namespace tensor_ops {
    // Join a sequence of arrays along an existing axis.
    Tensor concatenate(const std::vector<Tensor>& tensors, int axis) {
        // if concatenating a single tensor, just return the same tensor
        if (tensors.size() == 1) return tensors[0];

        if (tensors.size() > 0 && axis < 0) {
            axis = tensors[0].ndim() + axis;
        }

        std::vector<Array> arrays;
        arrays.reserve(tensors.size());
        bool constant = true;
        for (auto& t : tensors) {
            arrays.emplace_back(t.w);
            constant = constant & t.constant;
        }

        Tensor out(op::concatenate(arrays, axis));
        out.constant = constant;

        if (graph::backprop_enabled()) {
            graph::emplace_back([out, tensors, axis]() {
                int so_far = 0;
                for (auto& tensor : tensors) {
                    MAYBE_GRAD(tensor) <<= out.dw.pluck_axis(
                        axis,
                        Slice(so_far, so_far + tensor.shape()[axis])
                    );
                    so_far += tensor.shape()[axis];
                }
            });
        }
        return out;
    }
    // Join a sequence of arrays along their last axis.
    Tensor hstack(const std::vector<Tensor>& tensors) {
        return concatenate(tensors, -1);
    }
    // Stack arrays in sequence vertically (row wise).
    Tensor vstack(const std::vector<Tensor>& tensors) {
        return concatenate(tensors, 0);
    }
}
