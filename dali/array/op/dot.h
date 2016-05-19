#ifndef DALI_ARRAY_LAZY_DOT_H
#define DALI_ARRAY_LAZY_DOT_H

#include "dali/array/array.h"

namespace op {
    AssignableArray dot(const Array& a, const Array& b);
    AssignableArray vector_dot(const Array& a, const Array& b);
}

#endif  // DALI_ARRAY_LAZY_DOT_H
