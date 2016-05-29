#include "unary.h"

#include "dali/array/array.h"
#include "dali/array/lazy/unary.h"

namespace op {
    AssignableArray identity(const Array& x, const bool& always_copy) {
        if (always_copy) return lazy::identity(x);

        return AssignableArray([x](Array& out, const OPERATOR_T& operator_t) {
            // if always_copy is false and we are using operator= we can check
            // whether the output is either identical to the input (out == x)
            // or if out is a new array (out.is_stateless). In these cases no
            // computation is needed, and we can simply copy or transfer state
            // from x to out.
            // if these conditions are not met we fall back to a regular
            // assignment of memory from x to out with operator_t
            if (out == x && operator_t == OPERATOR_T_EQL) return;
            if (out.is_stateless() && operator_t == OPERATOR_T_EQL) {
                out = x;
                return;
            }
            ((AssignableArray)lazy::identity(x)).assign_to(out, operator_t);
        });
    }

    AssignableArray identity_or_swap(const Array& x) {
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
