#include "dali/array/op/elementwise.h"

#include "dali/array/array.h"
#include "dali/array/array_function.h"
#include "dali/array/TensorFunctions.h"

typedef Elementwise<TensorOps::op::sigmoid> Sigmoid;
Array sigmoid(const Array& x) { return Sigmoid::eval(x); }

typedef Elementwise<TensorOps::op::relu> Relu;
Array relu(const Array& x) { return Relu::eval(x); }

typedef Elementwise<TensorOps::op::log_or_zero> LogOrZero;
Array log_or_zero(const Array& x) { return LogOrZero::eval(x); }

typedef Elementwise<TensorOps::op::abs> Abs;
Array abs(const Array& x) { return Abs::eval(x); }

typedef Elementwise<TensorOps::op::sign> Sign;
Array sign(const Array& x) { return Sign::eval(x); }
