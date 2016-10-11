#include "unary_scalar.h"
#include "dali/tensor/op/unary.h"
#include "dali/array/op2/binary.h"
#include "dali/array/op2/unary.h"
#include "dali/array/op_overload/common.h"
#include "dali/array/op_overload/nonlazy.h"
#include "dali/tensor/tape.h"
#include "dali/tensor/tensor_macros.h"

namespace tensor_ops {
    ////////////////////////////////////////////////////////////////////////////
    //                              POW                                       //
    ////////////////////////////////////////////////////////////////////////////

    #define DALI_DEFINE_TENSOR_POW_OP(ARG1, ARG2) \
        Tensor scalar_pow(ARG1, ARG2) { \
            if (std::abs(scalar + 1) < 1e-9) { \
                return scalar_div(1.0, t); \
            } else if (std::abs((double)scalar - 0)   < 1e-9) { \
                return Tensor::fill(1.0, t.shape(), t.dtype(), t.preferred_device()); \
            } else if (std::abs((double)scalar - 0.5) < 1e-9) { \
                return tensor_ops::sqrt(t); \
            } else if (std::abs((double)scalar - 1.0) < 1e-9) { \
                return t; \
            } else if (std::abs((double)scalar - 2.0) < 1e-9) { \
                return tensor_ops::square(t); \
            } \
            Tensor out(op::pow(t.w, scalar)); \
            if (graph::backprop_enabled()) \
                graph::emplace_back([t, out, scalar]() mutable { \
                    MAYBE_GRAD(t) += scalar * op::pow(t.w, scalar - 1) * out.dw; \
                }); \
            return out; \
        }

    DALI_DEFINE_TENSOR_POW_OP(const Tensor& t, const double& scalar)
    DALI_DEFINE_TENSOR_POW_OP(const Tensor& t, const float& scalar)
    DALI_DEFINE_TENSOR_POW_OP(const Tensor& t, const int& scalar)

    #define DALI_DEFINE_TENSOR_RPOW_OP(ARG1, ARG2) \
    Tensor scalar_pow(ARG1, ARG2) { \
        Tensor out(op::pow(scalar, t.w)); \
        if (graph::backprop_enabled()) \
            graph::emplace_back([t, out, scalar]() mutable { \
                typedef std::remove_reference<decltype(scalar)>::type scalar_t; \
                MAYBE_GRAD(t) += op::log_or_zero(scalar) * out.w * out.dw; \
            }); \
        return out; \
    }

    DALI_DEFINE_TENSOR_RPOW_OP(const double& scalar, const Tensor& t)
    DALI_DEFINE_TENSOR_RPOW_OP(const float& scalar,  const Tensor& t)
    DALI_DEFINE_TENSOR_RPOW_OP(const int& scalar,    const Tensor& t)

}  // namespace tensor_ops
