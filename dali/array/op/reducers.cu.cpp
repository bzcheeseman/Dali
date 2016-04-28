#include "reducers.h"

#include "dali/array/lazy/reducers.h"

namespace op {
    AssignableArray sum_all(const Array& x) {
        return lazy::sum_all(x).as_assignable();
    }

    AssignableArray mean_all(const Array& x) {
        return lazy::sum_all(x).as_assignable();
    }
}; // namespace op
