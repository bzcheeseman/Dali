#define DALI_USE_LAZY 0
#include "op.h"

AssignableArray operator+(const Array& left, const Array& right) {
    return add(left,right);
}

AssignableArray operator+(const Array& left, const double& right) {
    return scalar_add(left, right);
}

Array& operator+=(Array& left, const double& right) {
    left = scalar_add(left, right);
    return left;
}

Array& operator+=(Array& left, const float& right) {
    left = scalar_add(left, right);
    return left;
}

Array& operator+=(Array& left, const int& right) {
    left = scalar_add(left, right);
    return left;
}

AssignableArray operator-(const Array& left, const Array& right) {
    return sub(left,right);
}

AssignableArray operator-(const Array& left, const double& right) {
    return scalar_add(left, -right);
}

AssignableArray operator*(const Array& left, const Array& right) {
    return eltmul(left,right);
}

AssignableArray operator/(const Array& left, const Array& right) {
    return eltdiv(left,right);
}
