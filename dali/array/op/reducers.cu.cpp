#include "reducers.h"

#include "dali/array/lazy/reducers.h"
#define DALI_USE_LAZY 1
#include "dali/array/op.h"

namespace op {
    AssignableArray sum(const Array& x) {
        return lazy::sum(x);
    }

    AssignableArray mean(const Array& x) {
        return lazy::sum(x / x.number_of_elements());
    }

    AssignableArray L2_norm(const Array& x) {
        return AssignableArray([x](Array& out, const OPERATOR_T& operator_t) {
            Array temp = lazy::sum(lazy::square(x));
            lazy::eval(lazy::sqrt(temp)).assign_to(out, operator_t);
        });
    }

    AssignableArray L2_norm(const Array& x, const int& axis) {
        return AssignableArray([x, axis](Array& out, const OPERATOR_T& operator_t) {
            Array temp = lazy::sum(lazy::square(x), axis);
            lazy::eval(lazy::sqrt(temp)).assign_to(out, operator_t);
        });
    }

    AssignableArray min(const Array& x) {
        return lazy::min(x);
    }

    AssignableArray max(const Array& x) {
        return lazy::max(x);
    }

    AssignableArray sum(const Array& x, const int& axis) {
        return lazy::sum(x, axis);
    }

    AssignableArray min(const Array& x, const int& axis) {
        return lazy::min(x, axis);
    }

    AssignableArray max(const Array& x, const int& axis) {
        return lazy::max(x, axis);
    }

    AssignableArray argmin(const Array& x, const int& axis) {
        return lazy::argmin(x, axis);
    }

    AssignableArray argmax(const Array& x, const int& axis) {
        return lazy::argmax(x, axis);
    }

    AssignableArray mean(const Array& x, const int& axis) {
    	auto reduced = lazy::sum(x, axis);
    	return reduced / x.shape()[axis]; // size of reduced axis
    }
}; // namespace op
