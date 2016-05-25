#ifndef DALI_ARRAY_OP_UNARY_H
#define DALI_ARRAY_OP_UNARY_H

class Array;
class AssignableArray;

namespace op {
    AssignableArray identity(const Array& x);
    AssignableArray sigmoid(const Array& x);
    AssignableArray tanh(const Array& x);
    AssignableArray relu(const Array& x);
    AssignableArray eltinv(const Array& x);
    AssignableArray exp(const Array& x);
    AssignableArray log(const Array& x);
    AssignableArray log_or_zero(const Array& x);
    AssignableArray abs(const Array& x);
    AssignableArray sign(const Array& x);
    AssignableArray square(const Array& x);
    AssignableArray softplus(const Array& x);
    AssignableArray cube(const Array& x);
    AssignableArray sqrt(const Array& x);
    AssignableArray rsqrt(const Array& x);
} // namespace op

#endif
