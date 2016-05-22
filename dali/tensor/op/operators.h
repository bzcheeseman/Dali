#ifndef DALI_TENSOR_OP_BINARY_OPERATORS_H
#define DALI_TENSOR_OP_BINARY_OPERATORS_H

#include "dali/tensor/tensor.h"

#define DALI_DECLARE_TENSOR_INTERACTION(SYMBOL)\
    Tensor operator SYMBOL (const Tensor& left, const Tensor& right);\

#define DALI_DECLARE_TENSOR_SCALAR_INTERACTION(SYMBOL)\
    Tensor operator SYMBOL (const Tensor& left, const double& right);\
    Tensor operator SYMBOL (const Tensor& left, const float& right );\
    Tensor operator SYMBOL (const Tensor& left, const int& right   );\
    Tensor operator SYMBOL (const double& left, const Tensor& right);\
    Tensor operator SYMBOL (const float& left,  const Tensor& right);\
    Tensor operator SYMBOL (const int& left,    const Tensor& right);\


DALI_DECLARE_TENSOR_INTERACTION(+);
DALI_DECLARE_TENSOR_INTERACTION(-);
DALI_DECLARE_TENSOR_INTERACTION(*);
DALI_DECLARE_TENSOR_INTERACTION(/);
DALI_DECLARE_TENSOR_INTERACTION(^);

DALI_DECLARE_TENSOR_SCALAR_INTERACTION(-);
DALI_DECLARE_TENSOR_SCALAR_INTERACTION(+);
DALI_DECLARE_TENSOR_SCALAR_INTERACTION(*);
DALI_DECLARE_TENSOR_SCALAR_INTERACTION(/);
DALI_DECLARE_TENSOR_SCALAR_INTERACTION(^);

#endif
