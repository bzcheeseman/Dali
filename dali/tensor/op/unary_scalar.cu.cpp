#include "unary.h"

#include "dali/array/functor.h"
#include "dali/array/lazy_op.h"
#include "dali/tensor/tape.h"
#include "dali/tensor/tensor_macros.h"

namespace tensor_ops {
    ////////////////////////////////////////////////////////////////////////////
    //                                 ADD                                    //
    ////////////////////////////////////////////////////////////////////////////
    #define DALI_DEFINE_TENSOR_ADD_OP(ARG1, ARG2)  \
        Tensor scalar_add(ARG1, ARG2) { \
            Tensor out(op::scalar_add(t.w, scalar));\
            if (graph::backprop_enabled()) \
                graph::emplace_back([t, out]() mutable { \
                    MAYBE_GRAD(t) += out.dw; \
                }); \
            return out; \
        }

    DALI_DEFINE_TENSOR_ADD_OP(const Tensor& t, const double& scalar)
    DALI_DEFINE_TENSOR_ADD_OP(const Tensor& t, const float& scalar)
    DALI_DEFINE_TENSOR_ADD_OP(const Tensor& t, const int& scalar)
    DALI_DEFINE_TENSOR_ADD_OP(const double& scalar,   const Tensor& t)
    DALI_DEFINE_TENSOR_ADD_OP(const float& scalar,    const Tensor& t)
    DALI_DEFINE_TENSOR_ADD_OP(const int& scalar,      const Tensor& t)

    ////////////////////////////////////////////////////////////////////////////
    //                                 SUB                                    //
    ////////////////////////////////////////////////////////////////////////////

    #define DALI_DEFINE_TENSOR_SUB_OP(ARG1, ARG2) \
        Tensor scalar_sub(ARG1, ARG2) { \
            Tensor out(op::scalar_sub(t.w, scalar)); \
            if (graph::backprop_enabled()) \
                graph::emplace_back([t, out]() mutable { \
                    MAYBE_GRAD(t) += out.dw; \
                }); \
            return out; \
        }

    DALI_DEFINE_TENSOR_SUB_OP(const Tensor& t, const double& scalar)
    DALI_DEFINE_TENSOR_SUB_OP(const Tensor& t, const float& scalar)
    DALI_DEFINE_TENSOR_SUB_OP(const Tensor& t, const int& scalar)

    #define DALI_DEFINE_TENSOR_RSUB_OP(ARG1, ARG2) \
        Tensor scalar_sub(ARG1, ARG2) { \
            Tensor out(op::scalar_sub(scalar, t.w)); \
             if (graph::backprop_enabled()) \
                graph::emplace_back([t, out]() mutable { \
                    MAYBE_GRAD(t) += -out.dw; \
                }); \
            return out; \
        } \

    DALI_DEFINE_TENSOR_RSUB_OP(const double& scalar,   const Tensor& t)
    DALI_DEFINE_TENSOR_RSUB_OP(const float& scalar,    const Tensor& t)
    DALI_DEFINE_TENSOR_RSUB_OP(const int& scalar,      const Tensor& t)


    ////////////////////////////////////////////////////////////////////////////
    //                              ELTMUL                                    //
    ////////////////////////////////////////////////////////////////////////////
    #define DALI_DEFINE_TENSOR_MUL_OP(ARG1, ARG2) \
        Tensor scalar_eltmul(ARG1, ARG2) { \
            Tensor out(op::scalar_mul(t.w, scalar)); \
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

    ////////////////////////////////////////////////////////////////////////////
    //                              ELTDIV                                    //
    ////////////////////////////////////////////////////////////////////////////

    #define DALI_DEFINE_TENSOR_DIV_OP(ARG1, ARG2) \
        Tensor scalar_div(ARG1, ARG2) { \
            Tensor out(op::scalar_div(t.w, scalar)); \
            if (graph::backprop_enabled()) \
                graph::emplace_back([t, out, scalar]() mutable { \
                    MAYBE_GRAD(t) += out.dw / scalar; \
                }); \
            return out; \
        }

    DALI_DEFINE_TENSOR_DIV_OP(const Tensor& t, const double& scalar)
    DALI_DEFINE_TENSOR_DIV_OP(const Tensor& t, const float& scalar)
    DALI_DEFINE_TENSOR_DIV_OP(const Tensor& t, const int& scalar)

    #define DALI_DEFINE_TENSOR_RDIV_OP(ARG1, ARG2) \
        Tensor scalar_div(ARG1, ARG2) { \
            Tensor out(op::scalar_div(scalar, t.w)); \
            if (graph::backprop_enabled()) \
                graph::emplace_back([t, out, scalar]() mutable { \
                    MAYBE_GRAD(t) += -scalar / lazy::square(out.dw); \
                }); \
            return out; \
        }

    DALI_DEFINE_TENSOR_RDIV_OP(const double& scalar, const Tensor& t)
    DALI_DEFINE_TENSOR_RDIV_OP(const float& scalar,  const Tensor& t)
    DALI_DEFINE_TENSOR_RDIV_OP(const int& scalar,    const Tensor& t)



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
            Tensor out(op::scalar_pow(t.w, scalar)); \
            if (graph::backprop_enabled()) \
                graph::emplace_back([t, out, scalar]() mutable { \
                    MAYBE_GRAD(t) += scalar * lazy::pow(t.w, scalar - 1) * out.dw; \
                }); \
            return out; \
        }

    DALI_DEFINE_TENSOR_POW_OP(const Tensor& t, const double& scalar)
    DALI_DEFINE_TENSOR_POW_OP(const Tensor& t, const float& scalar)
    DALI_DEFINE_TENSOR_POW_OP(const Tensor& t, const int& scalar)

    #define DALI_DEFINE_TENSOR_RPOW_OP(ARG1, ARG2) \
    Tensor scalar_pow(ARG1, ARG2) { \
        Tensor out(op::scalar_pow(scalar, t.w)); \
        if (graph::backprop_enabled()) \
            graph::emplace_back([t, out, scalar]() mutable { \
                MAYBE_GRAD(t) += functor::log_or_zero<decltype(scalar)>::Map(scalar) * out.w * out.dw; \
            }); \
        return out; \
    }

    DALI_DEFINE_TENSOR_RPOW_OP(const double& scalar, const Tensor& t)
    DALI_DEFINE_TENSOR_RPOW_OP(const float& scalar,  const Tensor& t)
    DALI_DEFINE_TENSOR_RPOW_OP(const int& scalar,    const Tensor& t)

}  // namespace tensor_ops
