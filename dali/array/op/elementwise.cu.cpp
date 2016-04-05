#include "dali/array/op/elementwise.h"

#include "dali/array/array.h"
#include "dali/array/array_function.h"
#include "dali/array/TensorFunctions.h"

typedef Elementwise<TensorOps::op::sigmoid> Sigmoid;
Array sigmoid(const Array& x) { return Sigmoid::eval(x); }
