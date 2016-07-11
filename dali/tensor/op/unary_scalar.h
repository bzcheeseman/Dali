#ifndef DALI_TENSOR_OP_UNARY_SCALAR_H
#define DALI_TENSOR_OP_UNARY_SCALAR_H

#include "dali/tensor/tensor.h"

namespace tensor_ops {

    #define DALI_DECLARE_TENSOR_SCALAR_OP(OPNAME) \
        Tensor OPNAME(const Tensor&,        const double& scalar); \
        Tensor OPNAME(const Tensor&,        const float& scalar ); \
        Tensor OPNAME(const Tensor&,        const int& scalar   ); \
        Tensor OPNAME(const double& scalar, const Tensor&); \
        Tensor OPNAME(const float& scalar,  const Tensor&); \
        Tensor OPNAME(const int& scalar,    const Tensor&);

    DALI_DECLARE_TENSOR_SCALAR_OP(scalar_add)
    DALI_DECLARE_TENSOR_SCALAR_OP(scalar_sub)
    DALI_DECLARE_TENSOR_SCALAR_OP(scalar_mul)
    DALI_DECLARE_TENSOR_SCALAR_OP(scalar_div)
    DALI_DECLARE_TENSOR_SCALAR_OP(scalar_pow)
}  // namespace tensor_ops

#endif  // DALI_TENSOR_OP_UNARY_SCALAR_H
