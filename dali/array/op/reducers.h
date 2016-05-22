#ifndef DALI_ARRAY_OP_REDUCERS_H
#define DALI_ARRAY_OP_REDUCERS_H

#include "dali/array/array.h"

namespace op {
    AssignableArray sum(const Array& x);
    AssignableArray L2_norm(const Array& x);
    AssignableArray L2_norm(const Array& x, const int& axis);
    AssignableArray mean(const Array& x);
    AssignableArray min(const Array& x);
    AssignableArray max(const Array& x);
    AssignableArray sum(const Array& x, const int& axis);
    AssignableArray mean(const Array& x, const int& axis);
    AssignableArray min(const Array& x, const int& axis);
    AssignableArray max(const Array& x, const int& axis);
    AssignableArray argmin(const Array& x, const int& axis);
    AssignableArray argmax(const Array& x, const int& axis);
}; // namespace op
#endif
