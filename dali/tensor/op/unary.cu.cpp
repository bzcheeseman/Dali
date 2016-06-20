#include "unary.h"

#include "dali/array/functor.h"
#include "dali/array/lazy_op.h"
#include "dali/tensor/tape.h"
#include "dali/tensor/tensor_macros.h"

namespace tensor_ops {
    DALI_DEFINE_UNARY_OP0(tanh, op::tanh, lazy::F<functor::dtanh>(out.w));
    DALI_DEFINE_UNARY_OP0(softplus, op::softplus, lazy::F<functor::softplus_backward>(t.w));
    DALI_DEFINE_UNARY_OP0(abs, op::abs, lazy::F<functor::sign>(t.w));
    DALI_DEFINE_UNARY_OP0(log, op::log, lazy::F<functor::inv>(t.w));
    DALI_DEFINE_UNARY_OP0(relu, op::relu, lazy::F<functor::relu_backward>(out.w));
    DALI_DEFINE_UNARY_OP0(exp, op::exp, out.w);
    DALI_DEFINE_UNARY_OP0(sigmoid, op::sigmoid, lazy::F<functor::dsigmoid>(out.w));
    DALI_DEFINE_UNARY_OP0(eltinv, op::eltinv, -lazy::square(out.w));
    DALI_DEFINE_UNARY_OP0(sqrt, op::sqrt, (0.5 / out.w));
}  // namespace tensor_ops
