#include "dali/tensor/op/unary_scalar.h"
#include "dali/array/op2/binary.h"
#include "dali/array/op2/unary.h"
#include "dali/array/op_overload/common.h"
#include "dali/array/op_overload/nonlazy.h"
#include "dali/tensor/tape.h"
#include "dali/tensor/tensor_macros.h"

namespace tensor_ops {
    ////////////////////////////////////////////////////////////////////////////
    //                              ELTMUL                                    //
    ////////////////////////////////////////////////////////////////////////////
    #define DALI_DEFINE_TENSOR_MUL_OP(ARG1, ARG2) \
        Tensor scalar_mul(ARG1, ARG2) { \
            Tensor out(op::eltmul(t.w, scalar)); \
            if (graph::backprop_enabled()) \
                graph::emplace_back([t, out, scalar]() mutable { \
                    MAYBE_GRAD(t) += scalar * out.dw; \
                }); \
            return out; \
        }

    DALI_DEFINE_TENSOR_MUL_OP(const Tensor& t, const double& scalar)
    DALI_DEFINE_TENSOR_MUL_OP(const Tensor& t, const float& scalar)
    DALI_DEFINE_TENSOR_MUL_OP(const Tensor& t, const int& scalar)
    DALI_DEFINE_TENSOR_MUL_OP(const double& scalar,   const Tensor& t)
    DALI_DEFINE_TENSOR_MUL_OP(const float& scalar,    const Tensor& t)
    DALI_DEFINE_TENSOR_MUL_OP(const int& scalar,      const Tensor& t)
}
