#include "unary.h"

#include "dali/array/lazy_op.h"

#include "dali/tensor/tape.h"
#include "dali/tensor/tensor_macros.h"

namespace tensor_ops {

    ////////////////////////////////////////////////////////////////////////////
    //                                 ADD                                    //
    ////////////////////////////////////////////////////////////////////////////
    #define DALI_DEFINE_TENSOR_ADD_OP(ARG1, ARG2)  \
        Tensor scalar_add(ARG1, ARG2) { \
            Tensor out(lazy::add(t.w, scalar));\
            if (graph::backprop_enabled()) \
                graph::emplace_back([t, out]() mutable { \
                    MAYBE_GRAD(t) <<= out.dw; \
                }); \
        }

    DALI_DEFINE_TENSOR_ADD_OP(const Tensor& t, double scalar)
    DALI_DEFINE_TENSOR_ADD_OP(const Tensor& t, float scalar)
    DALI_DEFINE_TENSOR_ADD_OP(const Tensor& t, int scalar)
    DALI_DEFINE_TENSOR_ADD_OP(double scalar,   const Tensor& t)
    DALI_DEFINE_TENSOR_ADD_OP(float scalar,    const Tensor& t)
    DALI_DEFINE_TENSOR_ADD_OP(int scalar,      const Tensor& t)

    ////////////////////////////////////////////////////////////////////////////
    //                                 SUB                                    //
    ////////////////////////////////////////////////////////////////////////////

    #define DALI_DEFINE_TENSOR_SUB_OP(ARG1, ARG2) \
        Tensor scalar_sub(ARG1, ARG2) { \
            Tensor out(lazy::sub(t.w, scalar)); \
            if (graph::backprop_enabled()) \
                graph::emplace_back([t, out]() mutable { \
                    MAYBE_GRAD(t) <<= out.dw; \
                }); \
        }

    DALI_DEFINE_TENSOR_SUB_OP(const Tensor& t, double scalar)
    DALI_DEFINE_TENSOR_SUB_OP(const Tensor& t, float scalar)
    DALI_DEFINE_TENSOR_SUB_OP(const Tensor& t, int scalar)

    #define DALI_DEFINE_TENSOR_RSUB_OP(ARG1, ARG2) \
        Tensor scalar_sub(ARG1, ARG2) { \
            Tensor out(lazy::sub(scalar, t.w)); \
             if (graph::backprop_enabled()) \
                graph::emplace_back([t, out]() mutable { \
                    MAYBE_GRAD(t) <<= -out.dw; \
                }); \
        } \

    DALI_DEFINE_TENSOR_RSUB_OP(double scalar,   const Tensor& t)
    DALI_DEFINE_TENSOR_RSUB_OP(float scalar,    const Tensor& t)
    DALI_DEFINE_TENSOR_RSUB_OP(int scalar,      const Tensor& t)


    ////////////////////////////////////////////////////////////////////////////
    //                              ELTMUL                                    //
    ////////////////////////////////////////////////////////////////////////////
    #define DALI_DEFINE_TENSOR_MUL_OP(ARG1, ARG2) \
        Tensor scalar_eltmul(ARG1, ARG2) { \
            Tensor out(lazy::eltmul(t.w, scalar)); \
            if (graph::backprop_enabled()) \
                graph::emplace_back([t, out, scalar]() mutable { \
                    MAYBE_GRAD(t) <<= scalar * out.dw; \
                }); \
        }

    DALI_DEFINE_TENSOR_MUL_OP(const Tensor& t, double scalar)
    DALI_DEFINE_TENSOR_MUL_OP(const Tensor& t, float scalar)
    DALI_DEFINE_TENSOR_MUL_OP(const Tensor& t, int scalar)
    DALI_DEFINE_TENSOR_MUL_OP(double scalar,   const Tensor& t)
    DALI_DEFINE_TENSOR_MUL_OP(float scalar,    const Tensor& t)
    DALI_DEFINE_TENSOR_MUL_OP(int scalar,      const Tensor& t)

    ////////////////////////////////////////////////////////////////////////////
    //                              ELTDIV                                    //
    ////////////////////////////////////////////////////////////////////////////

    #define DALI_DEFINE_TENSOR_DIV_OP(ARG1, ARG2) \
        Tensor scalar_eltdiv(ARG1, ARG2) { \
            Tensor out(lazy::eltdiv(t.w, scalar)); \
            if (graph::backprop_enabled()) \
                graph::emplace_back([t, out, scalar]() mutable { \
                    MAYBE_GRAD(t) <<= out.dw / scalar; \
                }); \
        }
        
    DALI_DEFINE_TENSOR_DIV_OP(const Tensor& t, double scalar)
    DALI_DEFINE_TENSOR_DIV_OP(const Tensor& t, float scalar)
    DALI_DEFINE_TENSOR_DIV_OP(const Tensor& t, int scalar)

    #define DALI_DEFINE_TENSOR_RDIV_OP(ARG1, ARG2) \
        Tensor scalar_eltdiv(ARG1, ARG2) { \
            Tensor out(lazy::eltdiv(scalar, t.w)); \
            if (graph::backprop_enabled()) \
                graph::emplace_back([t, out, scalar]() mutable { \
                    MAYBE_GRAD(t) <<= -scalar / lazy::square(out.dw); \
                }); \
        }

    DALI_DEFINE_TENSOR_RDIV_OP(double scalar,   const Tensor& t)
    DALI_DEFINE_TENSOR_RDIV_OP(float scalar,    const Tensor& t)
    DALI_DEFINE_TENSOR_RDIV_OP(int scalar,      const Tensor& t)
}  // namespace tensor_ops
