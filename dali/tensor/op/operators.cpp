#ifndef DALI_TENSOR_OP_BINARY_OPERATORS_H
#define DALI_TENSOR_OP_BINARY_OPERATORS_H

#include "dali/tensor/tensor.h"
#include "dali/tensor/op/unary_scalar.h"
#include "dali/tensor/op/binary.h"

#define DALI_DEFINE_TENSOR_INTERACTION(OPNAME, SYMBOL)\
    Tensor  operator SYMBOL    (const Tensor& left, const Tensor& right) {\
        return OPNAME(left, right);\
    } \
    Tensor& operator SYMBOL##= (Tensor& left,       const Tensor& right) {\
        return left = OPNAME(left, right);\
    }

#define DALI_DEFINE_TENSOR_SCALAR_INTERACTION(OPNAME, SYMBOL)\
    Tensor  operator SYMBOL    (const Tensor& left, const double& right) { return OPNAME(left,right); } \
    Tensor  operator SYMBOL    (const Tensor& left, const float& right ) { return OPNAME(left,right); } \
    Tensor  operator SYMBOL    (const Tensor& left, const int& right   ) { return OPNAME(left,right); } \
    Tensor  operator SYMBOL    (const double& left, const Tensor& right) { return OPNAME(left,right); } \
    Tensor  operator SYMBOL    (const float& left,  const Tensor& right) { return OPNAME(left,right); } \
    Tensor  operator SYMBOL    (const int& left,    const Tensor& right) { return OPNAME(left,right); } \
    Tensor& operator SYMBOL##= (Tensor& left,       const double& right) { return left = OPNAME(left,right); } \
    Tensor& operator SYMBOL##= (Tensor& left,       const float& right ) { return left = OPNAME(left,right); } \
    Tensor& operator SYMBOL##= (Tensor& left,       const int& right   ) { return left = OPNAME(left,right); } \

DALI_DEFINE_TENSOR_INTERACTION(tensor_ops::add,   +);
DALI_DEFINE_TENSOR_INTERACTION(tensor_ops::sub,   -);
DALI_DEFINE_TENSOR_INTERACTION(tensor_ops::eltmul,*);
DALI_DEFINE_TENSOR_INTERACTION(tensor_ops::eltdiv,/);
DALI_DEFINE_TENSOR_INTERACTION(tensor_ops::pow,   ^);

DALI_DEFINE_TENSOR_SCALAR_INTERACTION(tensor_ops::scalar_add, +);
DALI_DEFINE_TENSOR_SCALAR_INTERACTION(tensor_ops::scalar_sub, -);
DALI_DEFINE_TENSOR_SCALAR_INTERACTION(tensor_ops::scalar_mul,*);
DALI_DEFINE_TENSOR_SCALAR_INTERACTION(tensor_ops::scalar_div,/);
DALI_DEFINE_TENSOR_SCALAR_INTERACTION(tensor_ops::scalar_pow,   ^);
#endif
