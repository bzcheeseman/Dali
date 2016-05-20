#include "unary.h"

#include "dali/array/lazy/unary.h"
#include "dali/array/lazy/binary.h"


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

    AssignableArray relu(const Array& x) {
        return lazy::relu(x);
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

    AssignableArray sqrt(const Array& x) {
        return lazy::sqrt(x);
    }

    AssignableArray rsqrt(const Array& x) {
        return lazy::rsqrt(x);
    }

    AssignableArray scalar_add(const Array& x, const double& other) {
        return lazy::add(x, other);
    }
    AssignableArray scalar_add(const Array& x, const float& other) {
        return lazy::add(x, other);
    }
    AssignableArray scalar_add(const Array& x, const int& other) {
        return lazy::add(x, other);
    }
    AssignableArray scalar_sub(const Array& x, const double& other) {
        return lazy::sub(x, other);
    }
    AssignableArray scalar_sub(const Array& x, const float& other) {
        return lazy::sub(x, other);
    }
    AssignableArray scalar_sub(const Array& x, const int& other) {
        return lazy::sub(x, other);
    }
    AssignableArray scalar_mul(const Array& x, const double& other) {
        return lazy::eltmul(x, other);
    }
    AssignableArray scalar_mul(const Array& x, const float& other) {
        return lazy::eltmul(x, other);
    }
    AssignableArray scalar_mul(const Array& x, const int& other) {
        return lazy::eltmul(x, other);
    }
    AssignableArray scalar_div(const Array& x, const double& other) {
        return lazy::eltdiv(x, other);
    }
    AssignableArray scalar_div(const Array& x, const float& other) {
        return lazy::eltdiv(x, other);
    }
    AssignableArray scalar_div(const Array& x, const int& other) {
        return lazy::eltdiv(x, other);
    }

    AssignableArray pow(const Array& x, const double& other) {
        return lazy::pow(x, other);
    }
    AssignableArray pow(const Array& x, const float& other) {
        return lazy::pow(x, other);
    }
    AssignableArray pow(const Array& x, const int& other) {
        return lazy::pow(x, other);
    }
};
