#ifndef DALI_ARRAY_OP_BINARY_H
#define DALI_ARRAY_OP_BINARY_H

#include "dali/array/array.h"

namespace op {
    Array all_equals(Array left, Array right);
    Array add(Array left, Array right);
    Array subtract(Array left, Array right);
    Array eltmul(Array left, Array right);
    Array eltdiv(Array left, Array right);
    Array equals(Array left, Array right);
}

#endif
