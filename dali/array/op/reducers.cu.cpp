#include "reducers.h"

#include "dali/array/lazy/reducers.h"
#define DALI_USE_LAZY 1
#include "dali/array/op.h"

namespace op {
    AssignableArray sum_all(const Array& x) {
        return lazy::sum_all(x);
    }

    AssignableArray mean_all(const Array& x) {
        return lazy::sum_all(x / x.number_of_elements());
    }

    AssignableArray sum(const Array& x, const int& axis) {
        return lazy::sum_axis(x, axis);
    }

    AssignableArray mean(const Array& x, const int& axis) {
    	auto reduced = lazy::sum_axis(x, axis);
    	return reduced / x.shape()[axis]; // size of reduced axis
    }
}; // namespace op
