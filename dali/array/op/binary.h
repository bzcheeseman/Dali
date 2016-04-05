#ifndef DALI_ARRAY_OP_BINARY_H
#define DALI_ARRAY_OP_BINARY_H

#include "dali/array/array.h"

Array add(const Array& left, const Array& right);
Array eltmul(const Array& left, const Array& right);

Array operator+(const Array& left, const Array& right);
Array operator*(const Array& left, const Array& right);

#endif
