#ifndef DALI_TENSOR_OP_REDUCERS_H
#define DALI_TENSOR_OP_REDUCERS_H

#include "dali/tensor/tensor.h"

namespace tensor_ops {
	Tensor sum(const Tensor& tensor);
	Tensor mean(const Tensor& tensor);
}

#endif
