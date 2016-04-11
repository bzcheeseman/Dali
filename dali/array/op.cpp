#define DALI_USE_LAZY 0
#include "op.h"

AssignableArray operator+(const Array& left, const Array& right) {
    return add(left,right);
}

AssignableArray operator-(const Array& left, const Array& right) {
    return sub(left,right);
}

AssignableArray operator*(const Array& left, const Array& right) {
    return eltmul(left,right);
}

AssignableArray operator/(const Array& left, const Array& right) {
    return eltdiv(left,right);
}
