#include "unary.h"

#include "dali/array/op2/binary.h"
#include "dali/array/op2/unary.h"
#include "dali/array/op_overload/common.h"
#include "dali/array/op_overload/nonlazy.h"
#include "dali/tensor/tape.h"
#include "dali/tensor/tensor_macros.h"

namespace tensor_ops {
    DALI_DEFINE_UNARY_OP0(tanh, op::tanh, op::dtanh(out.w));
    DALI_DEFINE_UNARY_OP0(softplus, op::softplus, op::softplus_backward(t.w));
    DALI_DEFINE_UNARY_OP0(abs, op::abs, op::sign(t.w));
    DALI_DEFINE_UNARY_OP0(log, op::log, op::eltinv(t.w));
    DALI_DEFINE_UNARY_OP0(relu, op::relu, op::relu_backward(out.w));
    DALI_DEFINE_UNARY_OP0(exp, op::exp, out.w);
    DALI_DEFINE_UNARY_OP0(sigmoid, op::sigmoid, op::dsigmoid(out.w));
    DALI_DEFINE_UNARY_OP0(eltinv, op::eltinv, -op::square(out.w));
    DALI_DEFINE_UNARY_OP0(sqrt, op::sqrt, (0.5 / out.w));
}  // namespace tensor_ops
