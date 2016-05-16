#include "reducers.h"

#include "dali/array/lazy/reducers.h"
#define DALI_USE_LAZY 1
#include "dali/array/op.h"

namespace op {
    AssignableArray sum_all(const Array& x) {
        return lazy::sum_all(x).as_assignable();
    }

    AssignableArray mean_all(const Array& x) {
        return (lazy::sum_all(x / x.number_of_elements()) ).as_assignable();
    }

    AssignableArray sum(const Array& x, const int& axis) {
        return lazy::sum_axis(x, axis).as_assignable();
    }

    AssignableArray mean(const Array& x, const int& axis) {
    	auto reduced = lazy::sum_axis(x, axis);
    	auto normalized = reduced / x.shape()[axis]; // size of reduced axis
        return normalized.as_assignable();
    }
}; // namespace op
