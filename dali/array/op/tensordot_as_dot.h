#ifndef DALI_ARRAY_OP_TENSORDOT_AS_DOT_H
#define DALI_ARRAY_OP_TENSORDOT_AS_DOT_H

#include <vector>

void check_tensordot_reduce_axes(
        const std::vector<int>& operand_shape,
        char name,
        const std::vector<int>& reduce_axes,
        const bool& batched);

// Returns all the axes that are not being reduced.
std::vector<int> tensordot_nonreduced_axes(
        const int& ndim,
        const std::vector<int>& reduce_axes,
        const bool& batched);


namespace op {
    template<typename Container>
    using matmul_result_t = decltype(Container().dot(Container()));

    template<typename Container>
    matmul_result_t<Container> matrix_multiply_with_reshape(
            const Container& a,
            const Container& b,
            const std::vector<int>& out_shape,
            const std::vector<int>& out_shape_2d);

    template<typename Container>
    matmul_result_t<Container> tensordot_as_dot(
            const Container& a,
            const Container& b,
            const int& axis,
            bool batched);

    template<typename Container>
    matmul_result_t<Container> tensordot_as_dot(
            const Container& a,
            const Container& b,
            const std::vector<int>& a_reduce_axes,
            const std::vector<int>& b_reduce_axes,
            bool batched);
}

#include "dali/array/op/tensordot_as_dot-impl.h"

#endif  // DALI_ARRAY_OP_TENSORDOT_AS_DOT_H
