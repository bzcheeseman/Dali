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
    DALI_DEFINE_UNARY_OP0(square, op::square, t.w * 2.0);
    DALI_DEFINE_UNARY_OP0(cube, op::cube, op::square(out.w) * 3.0);
    DALI_DEFINE_UNARY_OP0(rsqrt, op::rsqrt, -0.5 * op::pow(t.w, -1.5));
    DALI_DEFINE_UNARY_OP1(eltmax, lower_bound, op::eltmax, op::greaterthanequal(t.w, lower_bound));
    DALI_DEFINE_UNARY_OP1(eltmin, upper_bound, op::eltmin, op::greaterthanequal(t.w, upper_bound));
    DALI_DEFINE_UNARY_OP1(steep_sigmoid, aggressiveness, op::steep_sigmoid, op::steep_sigmoid_backward(out.w, aggressiveness));
    DALI_DEFINE_UNARY_OP1(relu, upper_bound, op::clipped_relu, op::clipped_relu_backward(out.w, upper_bound));

    #define DALI_DEFINE_RELU_ACTIVATION(NAME, UPPER_BOUND)\
        Tensor NAME(const Tensor& t) {\
            return relu(t, UPPER_BOUND);\
        }

    DALI_DEFINE_RELU_ACTIVATION(relu100, 100.0);
    DALI_DEFINE_RELU_ACTIVATION(relu20, 20.0);
    DALI_DEFINE_RELU_ACTIVATION(relu6, 6.0);
    DALI_DEFINE_RELU_ACTIVATION(relu5, 5.0);

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

    DALI_DEFINE_TENSOR_POW_OP(const Tensor& t, const double& scalar);
    DALI_DEFINE_TENSOR_POW_OP(const Tensor& t, const float& scalar);
    DALI_DEFINE_TENSOR_POW_OP(const Tensor& t, const int& scalar);

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

    DALI_DEFINE_TENSOR_RPOW_OP(const double& scalar, const Tensor& t);
    DALI_DEFINE_TENSOR_RPOW_OP(const float& scalar,  const Tensor& t);
    DALI_DEFINE_TENSOR_RPOW_OP(const int& scalar,    const Tensor& t);

    #define DALI_DEFINE_TENSOR_ADD_OP(ARG1, ARG2)  \
        Tensor scalar_add(ARG1, ARG2) { \
            Tensor out(op::add(t.w, scalar));\
            if (graph::backprop_enabled()) \
                graph::emplace_back([t, out]() mutable { \
                    MAYBE_GRAD(t) += out.dw; \
                }); \
            return out; \
        }

    DALI_DEFINE_TENSOR_ADD_OP(const Tensor& t, const double& scalar);
    DALI_DEFINE_TENSOR_ADD_OP(const Tensor& t, const float& scalar);
    DALI_DEFINE_TENSOR_ADD_OP(const Tensor& t, const int& scalar);
    DALI_DEFINE_TENSOR_ADD_OP(const double& scalar,   const Tensor& t);
    DALI_DEFINE_TENSOR_ADD_OP(const float& scalar,    const Tensor& t);
    DALI_DEFINE_TENSOR_ADD_OP(const int& scalar,      const Tensor& t);

    #define DALI_DEFINE_TENSOR_SUB_OP(ARG1, ARG2) \
        Tensor scalar_sub(ARG1, ARG2) { \
            Tensor out(op::sub(t.w, scalar)); \
            if (graph::backprop_enabled()) \
                graph::emplace_back([t, out]() mutable { \
                    MAYBE_GRAD(t) += out.dw; \
                }); \
            return out; \
        }

    DALI_DEFINE_TENSOR_SUB_OP(const Tensor& t, const double& scalar);
    DALI_DEFINE_TENSOR_SUB_OP(const Tensor& t, const float& scalar);
    DALI_DEFINE_TENSOR_SUB_OP(const Tensor& t, const int& scalar);

    #define DALI_DEFINE_TENSOR_RSUB_OP(ARG1, ARG2) \
        Tensor scalar_sub(ARG1, ARG2) { \
            Tensor out(op::sub(scalar, t.w)); \
             if (graph::backprop_enabled()) \
                graph::emplace_back([t, out]() mutable { \
                    MAYBE_GRAD(t) += -out.dw; \
                }); \
            return out; \
        } \

    DALI_DEFINE_TENSOR_RSUB_OP(const double& scalar,   const Tensor& t);
    DALI_DEFINE_TENSOR_RSUB_OP(const float& scalar,    const Tensor& t);
    DALI_DEFINE_TENSOR_RSUB_OP(const int& scalar,      const Tensor& t);

    #define DALI_DEFINE_TENSOR_MUL_OP(ARG1, ARG2) \
        Tensor scalar_mul(ARG1, ARG2) { \
            Tensor out(op::eltmul(t.w, scalar)); \
            if (graph::backprop_enabled()) \
                graph::emplace_back([t, out, scalar]() mutable { \
                    MAYBE_GRAD(t) += scalar * out.dw; \
                }); \
            return out; \
        }

    DALI_DEFINE_TENSOR_MUL_OP(const Tensor& t, const double& scalar);
    DALI_DEFINE_TENSOR_MUL_OP(const Tensor& t, const float& scalar);
    DALI_DEFINE_TENSOR_MUL_OP(const Tensor& t, const int& scalar);
    DALI_DEFINE_TENSOR_MUL_OP(const double& scalar,   const Tensor& t);
    DALI_DEFINE_TENSOR_MUL_OP(const float& scalar,    const Tensor& t);
    DALI_DEFINE_TENSOR_MUL_OP(const int& scalar,      const Tensor& t);

    #define DALI_DEFINE_TENSOR_DIV_OP(ARG1, ARG2) \
        Tensor scalar_div(ARG1, ARG2) { \
            Tensor out(op::eltdiv(t.w, scalar)); \
            if (graph::backprop_enabled()) \
                graph::emplace_back([t, out, scalar]() mutable { \
                    MAYBE_GRAD(t) += out.dw / scalar; \
                }); \
            return out; \
        }

    DALI_DEFINE_TENSOR_DIV_OP(const Tensor& t, const double& scalar);
    DALI_DEFINE_TENSOR_DIV_OP(const Tensor& t, const float& scalar);
    DALI_DEFINE_TENSOR_DIV_OP(const Tensor& t, const int& scalar);

    #define DALI_DEFINE_TENSOR_RDIV_OP(ARG1, ARG2) \
        Tensor scalar_div(ARG1, ARG2) { \
            Tensor out(op::eltdiv(scalar, t.w)); \
            if (graph::backprop_enabled()) \
                graph::emplace_back([t, out, scalar]() mutable { \
                    MAYBE_GRAD(t) += -scalar / op::square(out.dw); \
                }); \
            return out; \
        }

    DALI_DEFINE_TENSOR_RDIV_OP(const double& scalar, const Tensor& t);
    DALI_DEFINE_TENSOR_RDIV_OP(const float& scalar,  const Tensor& t);
    DALI_DEFINE_TENSOR_RDIV_OP(const int& scalar,    const Tensor& t);
}  // namespace tensor_ops
