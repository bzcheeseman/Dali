#include "unary.h"

#include "dali/array/array.h"
#include "dali/array/lazy/unary.h"

namespace old_op {
    Assignable<Array> identity(const Array& x, const bool& always_copy) {
        if (always_copy) return lazy::identity(x);

        return Assignable<Array>([x](Array& out, const OPERATOR_T& operator_t) {
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
            ((Assignable<Array>)lazy::identity(x)).assign_to(out, operator_t);
        });
    }

    Assignable<Array> sigmoid(const Array& x) {
        return lazy::sigmoid(x);
    }

    Assignable<Array> tanh(const Array& x) {
        return lazy::tanh(x);
    }

    Assignable<Array> exp(const Array& x) {
        return lazy::exp(x);
    }

    Assignable<Array> softplus(const Array& x) {
        return lazy::softplus(x);
    }

    Assignable<Array> eltinv(const Array& x) {
        return lazy::eltinv(x);
    }

    Assignable<Array> relu(const Array& x) {
        return lazy::relu(x);
    }

    Assignable<Array> log(const Array& x) {
        return lazy::log(x);
    }

    Assignable<Array> log_or_zero(const Array& x) {
        return lazy::log_or_zero(x);
    }

    Assignable<Array> abs(const Array& x)  {
        return lazy::abs(x);
    }

    Assignable<Array> sign(const Array& x) {
        return lazy::sign(x);
    }

    Assignable<Array> square(const Array& x) {
        return lazy::square(x);
    }

    Assignable<Array> cube(const Array& x) {
        return lazy::cube(x);
    }

    Assignable<Array> sqrt(const Array& x) {
        return lazy::sqrt(x);
    }

    Assignable<Array> rsqrt(const Array& x) {
        return lazy::rsqrt(x);
    }
};
