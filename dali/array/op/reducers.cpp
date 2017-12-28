#include "reducers.h"
#include "dali/array/op/reducer_operation.h"
#include "dali/array/op/elementwise_operation.h"
#include "dali/array/op/binary.h"
#include "dali/array/op/unary.h"

namespace op {

    Array sum(const Array& x) {
        return all_reduce(x, "reducers::sum");
    }
    Array sum(const Array& x, const std::vector<int>& axes, bool keepdims) {
        return axis_reduce(x, "reducers::sum", axes, keepdims);
    }
    Array prod(const Array& x) {
        return all_reduce(x, "reducers::product");
    }
    Array prod(const Array& x, const std::vector<int>& axes, bool keepdims) {
        return axis_reduce(x, "reducers::product", axes, keepdims);
    }
    Array max(const Array& x) {
        return all_reduce(x, "reducers::maximum");
    }
    Array max(const Array& x, const std::vector<int>& axes, bool keepdims) {
        return axis_reduce(x, "reducers::maximum", axes, keepdims);
    }
    Array min(const Array& x) {
        return all_reduce(x, "reducers::minimum");
    }
    Array min(const Array& x, const std::vector<int>& axes, bool keepdims) {
        return axis_reduce(x, "reducers::minimum", axes, keepdims);
    }
    Array mean(const Array& x) {
        auto sum_op = all_reduce(x, "reducers::sum");
        if (sum_op.dtype() == DTYPE_INT32) sum_op = astype(sum_op, DTYPE_DOUBLE);
        return op::eltdiv(sum_op, op::identity(x.number_of_elements()));
    }
    Array mean(const Array& x, const std::vector<int>& axes, bool keepdims) {
        auto sum_op = axis_reduce(x, "reducers::sum", axes, keepdims);
        if (sum_op.dtype() == DTYPE_INT32) sum_op = astype(sum_op, DTYPE_DOUBLE);
        return op::eltdiv(sum_op, op::identity(x.number_of_elements() / sum_op.number_of_elements()));
    }
    Array L2_norm(const Array& x) {
        auto sum_op = all_reduce(op::square(x), "reducers::sum");
        if (sum_op.dtype() == DTYPE_INT32) sum_op = astype(sum_op, DTYPE_DOUBLE);
        return op::sqrt(sum_op);
    }
    Array L2_norm(const Array& x, const std::vector<int>& axes, bool keepdims) {
        auto sum_op = axis_reduce(op::square(x), "reducers::sum", axes, keepdims);
        if (sum_op.dtype() == DTYPE_INT32) sum_op = astype(sum_op, DTYPE_DOUBLE);
        return op::sqrt(sum_op);
    }
    Array argmax(const Array& x) {
        return argument_all_reduce(x, "reducers::maximum");
    }
    Array argmax(const Array& x, const int& axis) {
        return argument_axis_reduce(x, "reducers::maximum", axis);
    }
    Array argmin(const Array& x) {
        return argument_all_reduce(x, "reducers::minimum");
    }
    Array argmin(const Array& x, const int& axis) {
        return argument_axis_reduce(x, "reducers::minimum", axis);
    }
}
