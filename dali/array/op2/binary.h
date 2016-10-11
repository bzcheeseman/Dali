#ifndef DALI_ARRAY_OP2_BINARY_H
#define DALI_ARRAY_OP2_BINARY_H

#include "dali/array/op2/operation.h"

namespace op {
    Operation add(const Operation& a, const Operation& b);
    Assignable<Array> add(const std::vector<Array>& as);
    Operation sub(const Operation& a, const Operation& b);
    Operation eltmul(const Operation& left, const Operation& right);
    Operation eltdiv(const Operation& left, const Operation& right);
    Operation pow(const Operation& left, const Operation& right);
    Operation equals(const Operation& left, const Operation& right);
    Operation steep_sigmoid(const Operation& x, const Operation& aggressiveness);
    Operation steep_sigmoid_backward(const Operation& x, const Operation& aggressiveness);
    Operation clipped_relu(const Operation& x, const Operation& clipval);
    Operation clipped_relu_backward(const Operation& x, const Operation& clipval);
    Operation prelu(const Operation& x, const Operation& weights);
    Operation prelu_backward_weights(const Operation& a, const Operation& grad);
    Operation prelu_backward_inputs(const Operation& a, const Operation& weights);

    Operation lessthanequal(const Operation& a, const Operation& b);
    Operation greaterthanequal(const Operation& a, const Operation& b);
    Operation eltmax(const Operation& a, const Operation& b);
    Operation clip(const Operation& a, const Operation& b);
    Operation eltmin(const Operation& a, const Operation& b);
    Operation binary_cross_entropy(const Operation& a, const Operation& b);
    Operation binary_cross_entropy_grad(const Operation& a, const Operation& b);
}  // namespace op2

#endif  // DALI_ARRAY_OP2_BINARY_H
