#ifndef DALI_TENSOR_OP_UNARY_H
#define DALI_TENSOR_OP_UNARY_H

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
    DALI_DECLARE_TENSOR_SCALAR_OP(scalar_eltmul)
    DALI_DECLARE_TENSOR_SCALAR_OP(scalar_div)
    DALI_DECLARE_TENSOR_SCALAR_OP(scalar_pow)

    Tensor tanh(const Tensor&);
    Tensor softplus(const Tensor&);
    Tensor abs(const Tensor&);
    Tensor log(const Tensor&);
    // relu: relu(x) = max(x, 0);
    Tensor relu(const Tensor&);
    // clipped relu: relu(x ; clip) = max( min(clip, x), 0 );
    Tensor relu(const Tensor&, const double& upper_bound);
    Tensor exp(const Tensor&);
    Tensor sigmoid(const Tensor&);
    Tensor eltinv(const Tensor&);
    Tensor sqrt(const Tensor&);
    Tensor square(const Tensor&);
    Tensor cube(const Tensor&);
    Tensor rsqrt(const Tensor&);
    Tensor eltmax(const Tensor&, const double& lower_bound);
    Tensor eltmin(const Tensor&, const double& upper_bound);
    Tensor steep_sigmoid(const Tensor&, const double& agressiveness);

    Tensor relu100(const Tensor&);
    Tensor relu20(const Tensor&);
    Tensor relu6(const Tensor&);
    Tensor relu5(const Tensor&);
}  // namespace tensor_ops

#endif  // DALI_TENSOR_OP_UNARY_H
