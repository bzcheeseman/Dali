#ifndef DALI_TENSOR_OP_REDUCERS_H
#define DALI_TENSOR_OP_REDUCERS_H

#include "dali/tensor/tensor.h"

namespace tensor_ops {
    Tensor sum(const Tensor& tensor);
    Tensor mean(const Tensor& tensor);
    Tensor min(const Tensor& tensor);
    Tensor max(const Tensor& tensor);
    Tensor L2_norm(const Tensor& tensor);
    Tensor sum(const Tensor& tensor, const std::vector<int>& axes, bool keepdims=false);
    Tensor L2_norm(const Tensor& tensor, const std::vector<int>& axes, bool keepdims=false);
    Tensor mean(const Tensor& tensor, const std::vector<int>& axes, bool keepdims=false);
    Tensor min(const Tensor& tensor, const std::vector<int>& axes, bool keepdims=false);
    Tensor max(const Tensor& tensor, const std::vector<int>& axes, bool keepdims=false);
    Tensor argmax(const Tensor& t);
    Tensor argmin(const Tensor& t);
    Tensor argsort(const Tensor& t);
    Tensor argmax(const Tensor& t, int axis);
    Tensor argmin(const Tensor& t, int axis);
    Tensor argsort(const Tensor& t, int axis);
}

#endif
