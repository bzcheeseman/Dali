#ifndef DALI_ARRAY_OP_UNARY_H
#define DALI_ARRAY_OP_UNARY_H

#include "dali/array/array.h"

namespace op {
    Array identity(Array x);
    Array identity(int x);
    Array identity(float x);
    Array identity(double x);
}

#endif
