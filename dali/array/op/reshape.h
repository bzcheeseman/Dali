#ifndef DALI_ARRAY_OP_RESHAPE_H
#define DALI_ARRAY_OP_RESHAPE_H

#include <vector>
#include "dali/array/function/operator.h"

class Array;
class ArraySubtensor;
class AssignableArray;

namespace op {
    // Join a sequence of arrays along an existing axis.
    AssignableArray concatenate(const std::vector<Array>& arrays, int axis);
    // Join a sequence of arrays along their last axis.
    AssignableArray hstack(const std::vector<Array>& arrays);
    // Stack arrays in sequence vertically (row wise).
    AssignableArray vstack(const std::vector<Array>& arrays);
    // Pick indices from another array
    AssignableArray take(const Array& source, const Array& indices);
    // Pick indices from another array on each row
    // (equivalent to source[np.arange(0, n), indices] in numpy)
    AssignableArray take_from_rows(const Array& source, const Array& indices);

    template<OPERATOR_T operator_t>
    void assign_to_rows(const ArraySubtensor& destination,
                        const Array& assignable);
} // namespace op

#endif
