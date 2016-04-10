#ifndef DALI_ARRAY_OP_BINARY_H
#define DALI_ARRAY_OP_BINARY_H

#include "dali/array/array.h"
#include "dali/array/assignable_array.h"

AssignableArray add(const Array& left, const Array& right);
AssignableArray sub(const Array& left, const Array& right);
AssignableArray eltmul(const Array& left, const Array& right);
AssignableArray eltdiv(const Array& left, const Array& right);

AssignableArray operator+(const Array& left, const Array& right);
AssignableArray operator-(const Array& left, const Array& right);
AssignableArray operator*(const Array& left, const Array& right);
AssignableArray operator/(const Array& left, const Array& right);

#endif
