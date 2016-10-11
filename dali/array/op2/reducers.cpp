#include "reducers.h"
#include "dali/array/op2/operation.h"
#include "dali/array/op2/reducer_operation.h"
#include "dali/array/op2/elementwise_operation.h"
#include "dali/array/op2/binary.h"
#include "dali/array/op2/unary.h"

namespace op {

    Operation sum(const Operation& x) {
        return all_reduce(x, "reducers::sum");
    }
    Operation sum(const Operation& x, const std::vector<int>& axes) {
        return axis_reduce(x, "reducers::sum", axes);
    }
    Operation prod(const Operation& x) {
        return all_reduce(x, "reducers::product");
    }
    Operation prod(const Operation& x, const std::vector<int>& axes) {
        return axis_reduce(x, "reducers::product", axes);
    }
    Operation max(const Operation& x) {
        return all_reduce(x, "reducers::maximum");
    }
    Operation max(const Operation& x, const std::vector<int>& axes) {
        return axis_reduce(x, "reducers::maximum", axes);
    }
    Operation min(const Operation& x) {
        return all_reduce(x, "reducers::minimum");
    }
    Operation min(const Operation& x, const std::vector<int>& axes) {
        return axis_reduce(x, "reducers::minimum", axes);
    }
    Operation mean(const Operation& x) {
        auto sum_op = all_reduce(x, "reducers::sum");
        if (sum_op.dtype() == DTYPE_INT32) sum_op = astype(sum_op, DTYPE_DOUBLE);
        return op::eltdiv(sum_op, x.number_of_elements());
    }
    Operation mean(const Operation& x, const std::vector<int>& axes) {
        auto sum_op = axis_reduce(x, "reducers::sum", axes);
        if (sum_op.dtype() == DTYPE_INT32) sum_op = astype(sum_op, DTYPE_DOUBLE);
        return op::eltdiv(sum_op, x.number_of_elements() / sum_op.number_of_elements());
    }
    Operation L2_norm(const Operation& x) {
        auto sum_op = all_reduce(op::square(x), "reducers::sum");
        if (sum_op.dtype() == DTYPE_INT32) sum_op = astype(sum_op, DTYPE_DOUBLE);
        return op::sqrt(sum_op);
    }
    Operation L2_norm(const Operation& x, const std::vector<int>& axes) {
        auto sum_op = axis_reduce(op::square(x), "reducers::sum", axes);
        if (sum_op.dtype() == DTYPE_INT32) sum_op = astype(sum_op, DTYPE_DOUBLE);
        return op::sqrt(sum_op);
    }
    Operation argmax(const Operation& x) {
        return argument_all_reduce(x, "reducers::maximum");
    }
    Operation argmax(const Operation& x, const int& axis) {
        return argument_axis_reduce(x, "reducers::maximum", axis);
    }
    Operation argmin(const Operation& x) {
        return argument_all_reduce(x, "reducers::minimum");
    }
    Operation argmin(const Operation& x, const int& axis) {
        return argument_axis_reduce(x, "reducers::minimum", axis);
    }
}
