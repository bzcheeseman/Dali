#ifndef DALI_ARRAY_OP_DOT_H
#define DALI_ARRAY_OP_DOT_H

#include <vector>

class Array;
template<typename OutType>
class Assignable;

namespace op {
    Assignable<Array> dot(const Array& a, const Array& b);

    Assignable<Array> inner(const Array& a, const Array& b);

    Assignable<Array> matrixdot(const Array& a, const Array& b);

    Assignable<Array> matrix_vector_dot(const Array& a, const Array& b);

    Assignable<Array> tensordot(const Array& a, const Array& b, const int& axis);

    Assignable<Array> tensordot(const Array& a,
                              const Array& b,
                              const std::vector<int>& a_reduce_axes,
                              const std::vector<int>& b_reduce_axes);
}  // namespace op

#endif  // DALI_ARRAY_OP_DOT_H
