#include "binary.h"

#include "dali/array/lazy/binary.h"

namespace op {
    AssignableArray add(const Array& a, const Array& b) {
        return lazy::add(a, b).as_assignable();
    }

    AssignableArray sub(const Array& a, const Array& b) {
        return lazy::sub(a, b).as_assignable();
    }

    AssignableArray eltmul(const Array& a, const Array& b) {
        return lazy::eltmul(a, b).as_assignable();
    }

    AssignableArray eltdiv(const Array& a, const Array& b) {
        return lazy::eltdiv(a, b).as_assignable();
    }
}  // namespace op
