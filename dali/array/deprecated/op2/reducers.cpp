#include "reducers.h"
#include "dali/array/op2/reducer_operation.h"
#include "dali/array/op2/elementwise_operation.h"
#include "dali/array/op2/binary.h"
#include "dali/array/op2/unary.h"

namespace op {

    expression::ExpressionGraph sum(const expression::ExpressionGraph& x) {
        return all_reduce(x, "reducers::sum");
    }
    expression::ExpressionGraph sum(const expression::ExpressionGraph& x, const std::vector<int>& axes) {
        return axis_reduce(x, "reducers::sum", axes);
    }
    expression::ExpressionGraph prod(const expression::ExpressionGraph& x) {
        return all_reduce(x, "reducers::product");
    }
    expression::ExpressionGraph prod(const expression::ExpressionGraph& x, const std::vector<int>& axes) {
        return axis_reduce(x, "reducers::product", axes);
    }
    expression::ExpressionGraph max(const expression::ExpressionGraph& x) {
        return all_reduce(x, "reducers::maximum");
    }
    expression::ExpressionGraph max(const expression::ExpressionGraph& x, const std::vector<int>& axes) {
        return axis_reduce(x, "reducers::maximum", axes);
    }
    expression::ExpressionGraph min(const expression::ExpressionGraph& x) {
        return all_reduce(x, "reducers::minimum");
    }
    expression::ExpressionGraph min(const expression::ExpressionGraph& x, const std::vector<int>& axes) {
        return axis_reduce(x, "reducers::minimum", axes);
    }
    expression::ExpressionGraph mean(const expression::ExpressionGraph& x) {
        auto sum_op = all_reduce(x, "reducers::sum");
        if (sum_op.dtype() == DTYPE_INT32) sum_op = astype(sum_op, DTYPE_DOUBLE);
        return op::eltdiv(sum_op, x.number_of_elements());
    }
    expression::ExpressionGraph mean(const expression::ExpressionGraph& x, const std::vector<int>& axes) {
        auto sum_op = axis_reduce(x, "reducers::sum", axes);
        if (sum_op.dtype() == DTYPE_INT32) sum_op = astype(sum_op, DTYPE_DOUBLE);
        return op::eltdiv(sum_op, x.number_of_elements() / sum_op.number_of_elements());
    }
    expression::ExpressionGraph L2_norm(const expression::ExpressionGraph& x) {
        auto sum_op = all_reduce(op::square(x), "reducers::sum");
        if (sum_op.dtype() == DTYPE_INT32) sum_op = astype(sum_op, DTYPE_DOUBLE);
        return op::sqrt(sum_op);
    }
    expression::ExpressionGraph L2_norm(const expression::ExpressionGraph& x, const std::vector<int>& axes) {
        auto sum_op = axis_reduce(op::square(x), "reducers::sum", axes);
        if (sum_op.dtype() == DTYPE_INT32) sum_op = astype(sum_op, DTYPE_DOUBLE);
        return op::sqrt(sum_op);
    }
    expression::ExpressionGraph argmax(const expression::ExpressionGraph& x) {
        return argument_all_reduce(x, "reducers::maximum");
    }
    expression::ExpressionGraph argmax(const expression::ExpressionGraph& x, const int& axis) {
        return argument_axis_reduce(x, "reducers::maximum", axis);
    }
    expression::ExpressionGraph argmin(const expression::ExpressionGraph& x) {
        return argument_all_reduce(x, "reducers::minimum");
    }
    expression::ExpressionGraph argmin(const expression::ExpressionGraph& x, const int& axis) {
        return argument_axis_reduce(x, "reducers::minimum", axis);
    }
}
