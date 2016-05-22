#ifndef DALI_TENSOR_OP_COMPOSITE_H
#define DALI_TENSOR_OP_COMPOSITE_H

#include "dali/tensor/tensor.h"

namespace tensor_ops {

    Tensor quadratic_form(const Tensor& left, const Tensor& middle, const Tensor& right);

} // namespace tensor_ops

#endif
