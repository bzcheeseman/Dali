#ifndef DALI_ARRAY_OP_ELEMENTWISE_H
#define DALI_ARRAY_OP_ELEMENTWISE_H

#include "dali/array/array.h"

Array sigmoid(const Array& x);
Array relu(const Array& x);
Array log_or_zero(const Array& x);
Array abs(const Array& x);
Array sign(const Array& x);

#endif
