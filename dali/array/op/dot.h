#ifndef DALI_ARRAY_OP_DOT_H
#define DALI_ARRAY_OP_DOT_H

#include "dali/array/array.h"

namespace op {
    AssignableArray dot(const Array& a, const Array& b);

    AssignableArray outer(const Array& a, const Array& b);

    AssignableArray vectordot(const Array& a, const Array& b);

    AssignableArray matrixdot(const Array& a, const Array& b);

    AssignableArray matrix_vector_dot(const Array& a, const Array& b);

    AssignableArray tensordot(const Array& a, const Array& b, const int& axis);

    AssignableArray tensordot(const Array& a,
                              const Array& b,
                              const std::vector<int>& a_reduce_axes,
                              const std::vector<int>& b_reduce_axes);
}

#endif  // DALI_ARRAY_OP_DOT_H
