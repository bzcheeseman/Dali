#include "sgd.h"

#include "dali/array/op2/binary.h"
#include "dali/array/op2/unary.h"
#include "dali/array/op_overload/common.h"
#include "dali/array/op_overload/nonlazy.h"

namespace tensor_ops {
    void sgd_update(Tensor& param,
                    const double& step_size) {
        param.w -= step_size * param.dw;
    }
}  // namespace tensor_ops
