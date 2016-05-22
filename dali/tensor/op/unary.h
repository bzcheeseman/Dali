#ifndef DALI_TENSOR_OP_UNARY_H
#define DALI_TENSOR_OP_UNARY_H

#include "dali/tensor/tensor.h"

namespace tensor_ops {

    #define DALI_DECLARE_TENSOR_SCALAR_OP(OPNAME) \
        Tensor OPNAME(const Tensor&, double scalar); \
        Tensor OPNAME(const Tensor&, float scalar ); \
        Tensor OPNAME(const Tensor&, int scalar   ); \
        Tensor OPNAME(double scalar, const Tensor&); \
        Tensor OPNAME(float scalar,  const Tensor&); \
        Tensor OPNAME(int scalar,    const Tensor&);

    DALI_DECLARE_TENSOR_SCALAR_OP(scalar_add)
    DALI_DECLARE_TENSOR_SCALAR_OP(scalar_sub)
    DALI_DECLARE_TENSOR_SCALAR_OP(scalar_eltmul)
    DALI_DECLARE_TENSOR_SCALAR_OP(scalar_eltdiv)

    Tensor pow(const Tensor&, const Tensor& exponent);
    Tensor dot(const Tensor&, const Tensor&);
}  // namespace tensor_ops

#endif  // DALI_TENSOR_OP_UNARY_H
