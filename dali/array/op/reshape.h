#ifndef DALI_ARRAY_OP_RESHAPE_H
#define DALI_ARRAY_OP_RESHAPE_H

#include <vector>
#include "dali/array/function/operator.h"

class Array;
class ArraySubtensor;
template<typename OutType>
class Assignable;

namespace op {
    // Join a sequence of arrays along an existing axis.
    Assignable<Array> concatenate(const std::vector<Array>& arrays, int axis);
    // Join a sequence of arrays along their last axis.
    Assignable<Array> hstack(const std::vector<Array>& arrays);
    // Stack arrays in sequence vertically (row wise).
    Assignable<Array> vstack(const std::vector<Array>& arrays);
    // Pick indices from another array
    Assignable<Array> take(const Array& source, const Array& indices);
    // Pick indices from another array on each row
    // (equivalent to source[np.arange(0, n), indices] in numpy)
    Assignable<Array> take_from_rows(const Array& source, const Array& indices);
} // namespace op

#endif
