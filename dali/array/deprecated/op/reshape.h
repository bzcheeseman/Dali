// #ifndef DALI_ARRAY_OP_RESHAPE_H
// #define DALI_ARRAY_OP_RESHAPE_H

// #include <vector>
// #include "dali/array/expression/operator.h"

// class Array;
// struct ArraySubtensor;
// struct ArrayGather;
// template<typename OutType>
// struct Assignable;

// namespace old_op {
//     // Join a sequence of arrays along an existing axis.
//     Assignable<Array> concatenate(const std::vector<Array>& arrays, int axis);
//     // Join a sequence of arrays along their last axis.
//     Assignable<Array> hstack(const std::vector<Array>& arrays);
//     // Stack arrays in sequence vertically (row wise).
//     Assignable<Array> vstack(const std::vector<Array>& arrays);
//     // Pick indices from another array
//     Assignable<Array> gather(const Array& source, const Array& indices);
//     // Pick indices from another array on each row
//     // (equivalent to source[np.arange(0, n), indices] in numpy)
//     Assignable<Array> gather_from_rows(const Array& source, const Array& indices);
// }  // namespace old_op

// namespace internal {
//     void assign_to_rows(const Array& source, ArraySubtensor* dst);
//     void assign_to_gather(const Array& source, ArrayGather* dst);
// }  // namespace internal

// #endif
