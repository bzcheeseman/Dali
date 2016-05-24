#include "binary.h"

#include "dali/array/array.h"
#include "dali/array/lazy/binary.h"

namespace op {
    AssignableArray add(const Array& a, const Array& b) {
        return lazy::add(a, b);
    }

    AssignableArray sub(const Array& a, const Array& b) {
        return lazy::sub(a, b);
    }

    AssignableArray eltmul(const Array& a, const Array& b) {
        return lazy::eltmul(a, b);
    }

    AssignableArray eltdiv(const Array& a, const Array& b) {
        return lazy::eltdiv(a, b);
    }

    AssignableArray pow(const Array& a, const Array& b) {
        return lazy::pow(a, b);
    }
}  // namespace op
