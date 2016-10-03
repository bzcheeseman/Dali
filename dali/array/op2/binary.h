#ifndef DALI_ARRAY_OP2_BINARY_H
#define DALI_ARRAY_OP2_BINARY_H

class Operation;

namespace op2 {
    Operation add(const Operation& a, const Operation& b);
    Operation sub(const Operation& a, const Operation& b);
    Operation eltmul(const Operation& left, const Operation& right);
    Operation eltdiv(const Operation& left, const Operation& right);
    Operation pow(const Operation& left, const Operation& right);
    Operation equals(const Operation& left, const Operation& right);
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
