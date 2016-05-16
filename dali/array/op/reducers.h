#ifndef DALI_ARRAY_OP_REDUCERS_H
#define DALI_ARRAY_OP_REDUCERS_H

#include "dali/array/array.h"

namespace op {
    AssignableArray sum_all(const Array& x);
    AssignableArray mean_all(const Array& x);
    AssignableArray sum(const Array& x, const int& axis);
    AssignableArray mean(const Array& x, const int& axis);
}; // namespace op
#endif
