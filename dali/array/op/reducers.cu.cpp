#include "reducers.h"
#include "dali/array/array.h"
#include "dali/array/lazy/reducers.h"
#define DALI_USE_LAZY 1
#include "dali/array/op.h"


namespace op {
    Assignable<Array> sum(const Array& x) {
        return lazy::sum(x);
    }

    Assignable<Array> product(const Array& x) {
        return lazy::product(x);
    }

    Assignable<Array> mean(const Array& x) {
        return lazy::sum(x / x.number_of_elements());
    }

    Assignable<Array> L2_norm(const Array& x) {
        return Assignable<Array>([x](Array& out, const OPERATOR_T& operator_t) {
            Array temp = lazy::sum(lazy::square(x));
            lazy::eval_as_array(lazy::sqrt(temp)).assign_to(out, operator_t);
        });
    }

    Assignable<Array> L2_norm(const Array& x, const int& axis) {
        return Assignable<Array>([x, axis](Array& out, const OPERATOR_T& operator_t) {
            Array temp = lazy::sum(lazy::square(x), axis);
            lazy::eval_as_array(lazy::sqrt(temp)).assign_to(out, operator_t);
        });
    }

    Assignable<Array> min(const Array& x) {
        return lazy::min(x);
    }

    Assignable<Array> max(const Array& x) {
        return lazy::max(x);
    }

    Assignable<Array> sum(const Array& x, const int& axis) {
        return lazy::sum(x, axis);
    }

    Assignable<Array> product(const Array& x, const int& axis) {
        return lazy::product(x, axis);
    }

    Assignable<Array> min(const Array& x, const int& axis) {
        return lazy::min(x, axis);
    }

    Assignable<Array> max(const Array& x, const int& axis) {
        return lazy::max(x, axis);
    }

    Assignable<Array> argmin(const Array& x, const int& axis) {
        return lazy::argmin(x, axis);
    }

    Assignable<Array> argmax(const Array& x, const int& axis) {
        return lazy::argmax(x, axis);
    }

    Assignable<Array> mean(const Array& x, const int& axis) {
    	auto reduced = lazy::sum(x, axis);
    	return reduced / x.shape()[axis]; // size of reduced axis
    }
}; // namespace op
