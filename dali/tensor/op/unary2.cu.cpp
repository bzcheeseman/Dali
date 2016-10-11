#include "dali/tensor/op/unary.h"
#include "dali/array/op2/binary.h"
#include "dali/array/op2/unary.h"
#include "dali/array/op_overload/common.h"
#include "dali/array/op_overload/nonlazy.h"
#include "dali/tensor/tape.h"
#include "dali/tensor/tensor_macros.h"

namespace tensor_ops {
    DALI_DEFINE_UNARY_OP0(square, op::square, t.w * 2.0);
    DALI_DEFINE_UNARY_OP0(cube, op::cube, op::square(out.w) * 3.0);
    DALI_DEFINE_UNARY_OP0(rsqrt, op::rsqrt, -0.5 * op::pow(t.w, -1.5));

    DALI_DEFINE_UNARY_OP1(eltmax, lower_bound, op::eltmax, op::greaterthanequal(t.w, lower_bound));
    DALI_DEFINE_UNARY_OP1(eltmin, upper_bound, op::eltmin, op::greaterthanequal(t.w, upper_bound));
    DALI_DEFINE_UNARY_OP1(steep_sigmoid, aggressiveness, op::steep_sigmoid, op::steep_sigmoid_backward(out.w, aggressiveness));
    DALI_DEFINE_UNARY_OP1(relu, upper_bound, op::clipped_relu, op::clipped_relu_backward(out.w, upper_bound));
}  // namespace tensor_ops
