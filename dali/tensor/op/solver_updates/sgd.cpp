#include "sgd.h"

#include "dali/array/op/binary.h"
#include "dali/array/op/unary.h"

namespace tensor_ops {
    void sgd_update(Tensor& param,
                    const double& step_size) {
        param.w -= step_size * param.dw;
    }
}  // namespace tensor_ops
