#include "dali/tensor/op/unary.h"

#include "dali/array/functor.h"
#include "dali/array/lazy_op.h"
#include "dali/tensor/tape.h"
#include "dali/tensor/tensor_macros.h"

namespace tensor_ops {
    DALI_DEFINE_UNARY_OP0(square, op::square, t.w * 2.0);
    DALI_DEFINE_UNARY_OP0(cube, op::cube, lazy::square(out.w) * 3.0);
    DALI_DEFINE_UNARY_OP0(rsqrt, op::rsqrt, -0.5 * lazy::pow(t.w, -1.5));

    DALI_DEFINE_UNARY_OP1(eltmax, lower_bound, functor::max_scalar,
            lazy::F<functor::greaterthanequal>(t.w, lower_bound));
    DALI_DEFINE_UNARY_OP1(eltmin, upper_bound, functor::min_scalar,
            lazy::F<functor::greaterthanequal>(t.w, upper_bound));
    DALI_DEFINE_UNARY_OP1(steep_sigmoid, aggressiveness, functor::steep_sigmoid,
            lazy::F<functor::steep_sigmoid_backward>(out.w, aggressiveness));
    DALI_DEFINE_UNARY_OP1(relu, upper_bound, functor::clipped_relu,
            lazy::F<functor::clipped_relu_backward>(out.w, upper_bound));
}  // namespace tensor_ops
