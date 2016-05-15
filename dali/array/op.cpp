#define DALI_USE_LAZY 0
#include "op.h"

AssignableArray operator+(const Array& left, const Array& right) {
    return op::add(left, right);
}

AssignableArray operator+(const Array& left, const double& right) {
    return op::scalar_add(left, right);
}

Array& operator+=(Array& left, const Array& right) {
    left = op::add(left, right);
    return left;
}

Array& operator+=(Array& left, const double& right) {
    left = op::scalar_add(left, right);
    return left;
}

Array& operator+=(Array& left, const float& right) {
    left = op::scalar_add(left, right);
    return left;
}

Array& operator+=(Array& left, const int& right) {
    left = op::scalar_add(left, right);
    return left;
}

Array& operator-=(Array& left, const Array& right) {
    left = op::sub(left, right);
    return left;
}

Array& operator-=(Array& left, const double& right) {
    left = op::scalar_add(left, -right);
    return left;
}

Array& operator-=(Array& left, const float& right) {
    left = op::scalar_add(left, -right);
    return left;
}

Array& operator-=(Array& left, const int& right) {
    left = op::scalar_add(left, -right);
    return left;
}

Array& operator*=(Array& left, const Array& right) {
    left = op::eltmul(left, right);
    return left;
}

Array& operator*=(Array& left, const double& right) {
    left = op::scalar_mul(left, right);
    return left;
}

Array& operator*=(Array& left, const float& right) {
    left = op::scalar_mul(left, right);
    return left;
}

Array& operator*=(Array& left, const int& right) {
    left = op::scalar_mul(left, right);
    return left;
}

Array& operator/=(Array& left, const Array& right) {
    left = op::eltdiv(left, right);
    return left;
}

Array& operator/=(Array& left, const double& right) {
    left = op::scalar_div(left, right);
    return left;
}

Array& operator/=(Array& left, const float& right) {
    left = op::scalar_div(left, right);
    return left;
}

Array& operator/=(Array& left, const int& right) {
    left = op::scalar_div(left, right);
    return left;
}

Array& operator<<=(Array& left, const Array& right) {
    left <<= op::identity(right);
    return left;
}

AssignableArray operator-(const Array& left, const Array& right) {
    return op::sub(left,right);
}

AssignableArray operator-(const Array& left, const double& right) {
    return op::scalar_add(left, -right);
}

AssignableArray operator*(const Array& left, const Array& right) {
    return op::eltmul(left,right);
}

AssignableArray operator/(const Array& left, const Array& right) {
    return op::eltdiv(left,right);
}
