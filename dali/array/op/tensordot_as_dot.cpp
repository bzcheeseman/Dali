#include "tensordot_as_dot.h"

void check_tensordot_reduce_axes(
        const std::vector<int>& operand_shape,
        char name,
        const std::vector<int>& reduce_axes,
        const bool& batched) {
    // Do not reduce over more dimensions than operand_shape.size().
    ASSERT2(reduce_axes.size() <= operand_shape.size(),
        utils::MS() << "length of argument " << name << "_reduce_axes "
                    << "should be less than the dimensions of " << name
                    << " (" << name << ".ndim()=" << operand_shape
                    << ", " << name << "_reduce_axes.size()="
                    << reduce_axes.size() << ")."
    );
    // all reduction axes must be less than operand_shape.size()
    auto max_reduce_dim = std::max_element(
        reduce_axes.begin(), reduce_axes.end()
    );
    ASSERT2(reduce_axes.size() == 0 || (*max_reduce_dim) < operand_shape.size(),
        utils::MS() << name << "_reduce_axes contains reduction dimensions "
                    << " that are greater than or equal to "
                    << name << ".ndim() ("
                    << name << ".ndim()=" << operand_shape.size()
                    << ", and found max(" << name << "_reduce_axes)="
                    << *max_reduce_dim << ")."
    );
    if (batched) {
        auto find_iter = std::find(reduce_axes.begin(), reduce_axes.end(), 0);
        bool reducing_over_dim0 = find_iter != reduce_axes.end();
        ASSERT2(reducing_over_dim0,
            utils::MS() << "axes to sum over must not contain the batch axis "
                        << "(" << name << "_reduce_axes="
                        << reduce_axes << ")."
        );
    }
}

// Returns all the axes that are not being reduced.
std::vector<int> tensordot_nonreduced_axes(
        const int& ndim,
        const std::vector<int>& reduce_axes,
        const bool& batched) {
    std::vector<int> other_axes;
    for (int x = 0; x < ndim; x++) {
        // when batched, 0 is always kept
        // as leading dim, and thus will not
        // be dimshuffled
        if (batched && x == 0) {
            continue;
        }
        bool not_in_reduce_axes = (
            std::find(
                reduce_axes.begin(),
                reduce_axes.end(),
                x
            ) == reduce_axes.end()
        );

        if (not_in_reduce_axes) {
            other_axes.emplace_back(x);
        }
    }
    return other_axes;
}
