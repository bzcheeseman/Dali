#ifndef DALI_ARRAY_OP_BINARY_H
#define DALI_ARRAY_OP_BINARY_H

#include "dali/array/array.h"

namespace op {
    AssignableArray add(const Array& left, const Array& right);
    AssignableArray sub(const Array& left, const Array& right);
    AssignableArray eltmul(const Array& left, const Array& right);
    AssignableArray eltdiv(const Array& left, const Array& right);
    AssignableArray pow(const Array& left, const Array& right);
}  // namespace op
#endif
