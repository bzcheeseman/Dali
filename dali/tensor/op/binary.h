#ifndef DALI_TENSOR_OP_BINARY_H
#define DALI_TENSOR_OP_BINARY_H

#include "dali/tensor/tensor.h"

namespace tensor_ops {
    Tensor add(Tensor, Tensor);
    Tensor sub(Tensor, Tensor);
    Tensor eltmul(Tensor, Tensor);
    Tensor eltdiv(Tensor, Tensor);
    Tensor pow(Tensor, Tensor);
}

#endif
