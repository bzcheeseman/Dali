#include "unary.h"

#include "dali/array/lazy/unary.h"
#include "dali/array/lazy/binary.h"


namespace op {
    AssignableArray identity(const Array& x) {
        return lazy::identity(x).as_assignable();
    }

    AssignableArray sigmoid(const Array& x) {
        return lazy::sigmoid(x).as_assignable();
    }

    AssignableArray tanh(const Array& x) {
        return lazy::tanh(x).as_assignable();
    }

    AssignableArray relu(const Array& x) {
        return lazy::relu(x).as_assignable();
    }

    AssignableArray log_or_zero(const Array& x) {
        return lazy::log_or_zero(x).as_assignable();
    }

    AssignableArray abs(const Array& x)  {
        return lazy::abs(x).as_assignable();
    }

    AssignableArray sign(const Array& x) {
        return lazy::sign(x).as_assignable();
    }

    AssignableArray scalar_add(const Array& x, const double& other) {
        return lazy::add(x, other).as_assignable();
    }
    AssignableArray scalar_add(const Array& x, const float& other) {
        return lazy::add(x, other).as_assignable();
    }
    AssignableArray scalar_add(const Array& x, const int& other) {
        return lazy::add(x, other).as_assignable();
    }
    AssignableArray scalar_mul(const Array& x, const double& other) {
        return lazy::eltmul(x, other).as_assignable();
    }
    AssignableArray scalar_mul(const Array& x, const float& other) {
        return lazy::eltmul(x, other).as_assignable();
    }
    AssignableArray scalar_mul(const Array& x, const int& other) {
        return lazy::eltmul(x, other).as_assignable();
    }
    AssignableArray scalar_div(const Array& x, const double& other) {
        return lazy::eltdiv(x, other).as_assignable();
    }
    AssignableArray scalar_div(const Array& x, const float& other) {
        return lazy::eltdiv(x, other).as_assignable();
    }
    AssignableArray scalar_div(const Array& x, const int& other) {
        return lazy::eltdiv(x, other).as_assignable();
    }
};
