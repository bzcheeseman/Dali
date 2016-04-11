#ifndef DALI_ARRAY_OP_ELEMENTWISE_H
#define DALI_ARRAY_OP_ELEMENTWISE_H

#include "dali/array/array.h"

AssignableArray sigmoid(const Array& x);
AssignableArray tanh(const Array& x);
AssignableArray relu(const Array& x);
AssignableArray log_or_zero(const Array& x);
AssignableArray abs(const Array& x);
AssignableArray sign(const Array& x);

#endif
