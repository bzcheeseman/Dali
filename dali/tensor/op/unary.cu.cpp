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
                    MAYBE_GRAD(t) <<= out.dw; \
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
                    MAYBE_GRAD(t) <<= out.dw; \
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
                    MAYBE_GRAD(t) <<= -out.dw; \
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
                    MAYBE_GRAD(t) <<= scalar * out.dw; \
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
                    MAYBE_GRAD(t) <<= out.dw / scalar; \
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
                    MAYBE_GRAD(t) <<= -scalar / lazy::square(out.dw); \
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
                    MAYBE_GRAD(t) <<= scalar * lazy::pow(t.w, scalar - 1) * out.dw; \
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
                MAYBE_GRAD(t) <<= functor::log_or_zero<decltype(scalar)>::Map(scalar) * out.w * out.dw; \
            }); \
        return out; \
    }

    DALI_DEFINE_TENSOR_RPOW_OP(const double& scalar, const Tensor& t)
    DALI_DEFINE_TENSOR_RPOW_OP(const float& scalar,  const Tensor& t)
    DALI_DEFINE_TENSOR_RPOW_OP(const int& scalar,    const Tensor& t)

    ////////////////////////////////////////////////////////////////////////////
    //                              OTHERS                                    //
    ////////////////////////////////////////////////////////////////////////////

    #define DALI_UNARY_OP0(FUNCTION_NAME, FORWARD_OPNAME, BACKWARD_OPNAME) \
        Tensor FUNCTION_NAME(const Tensor& t) {\
            Tensor out(FORWARD_OPNAME(t.w));\
            if (graph::backprop_enabled() && !t.constant)\
                graph::emplace_back([t, out]() mutable {\
                    MAYBE_GRAD(t) <<= (BACKWARD_OPNAME) * out.dw;\
                });\
            return out;\
        }

    #define DALI_UNARY_OP1(FUNCTION_NAME, arg1, FORWARD_OPNAME, BACKWARD_OPNAME) \
        Tensor FUNCTION_NAME(const Tensor& t, const double& arg1) {\
            Tensor out(lazy::F<FORWARD_OPNAME>(t.w, arg1));\
            if (graph::backprop_enabled() && !t.constant)\
                graph::emplace_back([t, out, arg1]() mutable {\
                    MAYBE_GRAD(t) <<= (BACKWARD_OPNAME) * out.dw;\
                });\
            return out;\
        }
    DALI_UNARY_OP0(tanh, op::tanh, lazy::F<functor::dtanh>(out.w));
    DALI_UNARY_OP0(softplus, op::softplus, lazy::F<functor::softplus_backward>(t.w));
    DALI_UNARY_OP0(abs, op::abs, lazy::F<functor::sign>(t.w));
    DALI_UNARY_OP0(log, op::log, lazy::F<functor::inv>(t.w));
    DALI_UNARY_OP0(relu, op::relu, lazy::F<functor::relu_backward>(out.w));
    DALI_UNARY_OP0(exp, op::exp, out.w);
    DALI_UNARY_OP0(sigmoid, op::sigmoid, lazy::F<functor::dsigmoid>(out.w));
    DALI_UNARY_OP0(eltinv, op::eltinv, -lazy::square(out.w));
    DALI_UNARY_OP0(sqrt, op::sqrt, (0.5 / out.w));
    DALI_UNARY_OP0(square, op::square, t.w * 2.0);
    DALI_UNARY_OP0(cube, op::cube, lazy::square(out.w) * 3.0);
    DALI_UNARY_OP0(rsqrt, op::rsqrt, -0.5 * lazy::pow(t.w, -1.5));

    DALI_UNARY_OP1(eltmax, lower_bound, functor::max_scalar,
            lazy::F<functor::max_scalar_mask>(t.w, lower_bound));
    DALI_UNARY_OP1(eltmin, upper_bound, functor::min_scalar,
            lazy::F<functor::max_scalar_mask>(t.w, upper_bound));
    DALI_UNARY_OP1(steep_sigmoid, aggressiveness, functor::steep_sigmoid,
            lazy::F<functor::steep_sigmoid_backward>(out.w, aggressiveness));
    DALI_UNARY_OP1(relu, upper_bound, functor::clipped_relu,
            lazy::F<functor::clipped_relu_backward>(out.w, upper_bound));

    #define DALI_DEFINE_RELU_ACTIVATION(NAME, UPPER_BOUND)\
        Tensor NAME(const Tensor& t) {\
            return relu(t, UPPER_BOUND);\
        }

    DALI_DEFINE_RELU_ACTIVATION(relu100, 100.0);
    DALI_DEFINE_RELU_ACTIVATION(relu20, 20.0);
    DALI_DEFINE_RELU_ACTIVATION(relu6, 6.0);
    DALI_DEFINE_RELU_ACTIVATION(relu5, 5.0);

}  // namespace tensor_ops
