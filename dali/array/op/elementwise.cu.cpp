#include "dali/array/op/elementwise.h"

#include <iostream>

#include "dali/utils.h"
#include "dali/array/array.h"
#include "dali/array/array_function.h"
#include "dali/array/memory/device.h"
#include "dali/array/TensorFunctions.h"

using memory::Device;

typedef Elementwise<TensorOps::op::sigmoid> Sigmoid;
Array sigmoid(const Array& x) { return Sigmoid::eval(x); }
