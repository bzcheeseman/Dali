#include "dali/tensor/op/unary_scalar.h"

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
}
