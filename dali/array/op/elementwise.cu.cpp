#include "dali/array/op/elementwise.h"
#include "dali/array/array.h"
#include "dali/array/function/function.h"
#include "dali/array/TensorFunctions.h"
#include "dali/runtime_config.h"

#define DALI_USE_LAZY 1
#include "dali/array/op.h"

typedef Elementwise<TensorOps::op::sigmoid> Sigmoid;
AssignableArray sigmoid(const Array& x) { return Sigmoid::run(x); }

typedef Elementwise<TensorOps::op::tanh> Tanh;
AssignableArray tanh(const Array& x) { return Tanh::run(x); }

typedef Elementwise<TensorOps::op::relu> Relu;
AssignableArray relu(const Array& x) { return Relu::run(x); }

typedef Elementwise<TensorOps::op::log_or_zero> LogOrZero;
AssignableArray log_or_zero(const Array& x) { return LogOrZero::run(x); }

typedef Elementwise<TensorOps::op::abs> Abs;
AssignableArray abs(const Array& x)  { return Abs::run(x); }

typedef Elementwise<TensorOps::op::sign> Sign;
AssignableArray sign(const Array& x) { return Sign::run(x); }

AssignableArray scalar_add(const Array& x, const double& other) {
    return (x + other).as_assignable();
}
AssignableArray scalar_add(const Array& x, const float& other) {
    return (x + other).as_assignable();
}
AssignableArray scalar_add(const Array& x, const int& other) {
    return (x + other).as_assignable();
}
