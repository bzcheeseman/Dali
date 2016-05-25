#include "unary.h"

#include "dali/array/array.h"
#include "dali/array/lazy/unary.h"

namespace op {
    AssignableArray identity(const Array& x) {
        return lazy::identity(x);
    }

    AssignableArray sigmoid(const Array& x) {
        return lazy::sigmoid(x);
    }

    AssignableArray tanh(const Array& x) {
        return lazy::tanh(x);
    }

    AssignableArray exp(const Array& x) {
        return lazy::exp(x);
    }

    AssignableArray softplus(const Array& x) {
        return lazy::softplus(x);
    }

    AssignableArray eltinv(const Array& x) {
        return lazy::eltinv(x);
    }

    AssignableArray relu(const Array& x) {
        return lazy::relu(x);
    }

    AssignableArray log(const Array& x) {
        return lazy::log(x);
    }

    AssignableArray log_or_zero(const Array& x) {
        return lazy::log_or_zero(x);
    }

    AssignableArray abs(const Array& x)  {
        return lazy::abs(x);
    }

    AssignableArray sign(const Array& x) {
        return lazy::sign(x);
    }

    AssignableArray square(const Array& x) {
        return lazy::square(x);
    }

    AssignableArray cube(const Array& x) {
        return lazy::cube(x);
    }

    AssignableArray sqrt(const Array& x) {
        return lazy::sqrt(x);
    }

    AssignableArray rsqrt(const Array& x) {
        return lazy::rsqrt(x);
    }
};
