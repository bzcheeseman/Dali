#include "reducers.h"
#include "dali/tensor/tape.h"
#include "dali/array/op.h"
#include "dali/utils/print_utils.h"

namespace tensor_ops {
    Tensor sum(const Tensor& tensor) {
        if (tensor.number_of_elements() == 1) {
            auto out = tensor;
            out.w = tensor.w.reshape({});
            return out;
        } else {
            ELOG("yo");
            // TODO(jonathan, szymon) also makes sure that device
            // of input tensor is also used here
            Tensor out({}, initializer::empty(), tensor.dtype());
            ELOG("pre-yo");
            out.w = tensor.w.sum();
            ELOG("post-yo");

            if (graph::backprop_enabled() && !tensor.constant)
                graph::emplace_back([tensor, out]() mutable {
                    ELOG("yo grad");
                    tensor.dw += out.dw;
                    ELOG("post grad");
                });
            return out;
        }
    }
}
