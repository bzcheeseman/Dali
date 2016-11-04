#include "reducers.h"
#include "dali/array/op2/expression/expression.h"
#include "dali/array/op2/reducer_operation.h"
#include "dali/array/op2/elementwise_operation.h"
#include "dali/array/op2/binary.h"
#include "dali/array/op2/unary.h"

namespace op {

    Expression sum(const Expression& x) {
        return all_reduce(x, "reducers::sum");
    }
    Expression sum(const Expression& x, const std::vector<int>& axes) {
        return axis_reduce(x, "reducers::sum", axes);
    }
    Expression prod(const Expression& x) {
        return all_reduce(x, "reducers::product");
    }
    Expression prod(const Expression& x, const std::vector<int>& axes) {
        return axis_reduce(x, "reducers::product", axes);
    }
    Expression max(const Expression& x) {
        return all_reduce(x, "reducers::maximum");
    }
    Expression max(const Expression& x, const std::vector<int>& axes) {
        return axis_reduce(x, "reducers::maximum", axes);
    }
    Expression min(const Expression& x) {
        return all_reduce(x, "reducers::minimum");
    }
    Expression min(const Expression& x, const std::vector<int>& axes) {
        return axis_reduce(x, "reducers::minimum", axes);
    }
    Expression mean(const Expression& x) {
        auto sum_op = all_reduce(x, "reducers::sum");
        if (sum_op.dtype() == DTYPE_INT32) sum_op = astype(sum_op, DTYPE_DOUBLE);
        return op::eltdiv(sum_op, x.number_of_elements());
    }
    Expression mean(const Expression& x, const std::vector<int>& axes) {
        auto sum_op = axis_reduce(x, "reducers::sum", axes);
        if (sum_op.dtype() == DTYPE_INT32) sum_op = astype(sum_op, DTYPE_DOUBLE);
        return op::eltdiv(sum_op, x.number_of_elements() / sum_op.number_of_elements());
    }
    Expression L2_norm(const Expression& x) {
        auto sum_op = all_reduce(op::square(x), "reducers::sum");
        if (sum_op.dtype() == DTYPE_INT32) sum_op = astype(sum_op, DTYPE_DOUBLE);
        return op::sqrt(sum_op);
    }
    Expression L2_norm(const Expression& x, const std::vector<int>& axes) {
        auto sum_op = axis_reduce(op::square(x), "reducers::sum", axes);
        if (sum_op.dtype() == DTYPE_INT32) sum_op = astype(sum_op, DTYPE_DOUBLE);
        return op::sqrt(sum_op);
    }
    Expression argmax(const Expression& x) {
        return argument_all_reduce(x, "reducers::maximum");
    }
    Expression argmax(const Expression& x, const int& axis) {
        return argument_axis_reduce(x, "reducers::maximum", axis);
    }
    Expression argmin(const Expression& x) {
        return argument_all_reduce(x, "reducers::minimum");
    }
    Expression argmin(const Expression& x, const int& axis) {
        return argument_axis_reduce(x, "reducers::minimum", axis);
    }
}
