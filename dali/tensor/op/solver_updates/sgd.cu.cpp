#include "sgd.h"

#include "dali/array/lazy_op.h"

namespace tensor_ops {
    void sgd_update(Tensor& param,
                    const double& step_size) {
        param.w -= step_size * param.dw;
    }
}  // namespace tensor_ops
