#ifndef DALI_TENSOR_OP_REDUCERS_H
#define DALI_TENSOR_OP_REDUCERS_H

#include "dali/tensor/tensor.h"

namespace tensor_ops {
    Tensor sum(const Tensor& tensor);
    Tensor mean(const Tensor& tensor);
    Tensor min(const Tensor& tensor);
    Tensor max(const Tensor& tensor);
    Tensor sum(const Tensor& tensor, const int& axis);
    Tensor mean(const Tensor& tensor, const int& axis);
    Tensor min(const Tensor& tensor, const int& axis);
    Tensor max(const Tensor& tensor, const int& axis);
}

#endif
