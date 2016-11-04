#include "reducers.h"
#include "dali/array/op2/expression/expression.h"
#include "dali/array/op2/reducer_operation.h"
#include "dali/array/op2/elementwise_operation.h"
#include "dali/array/op2/binary.h"
#include "dali/array/op2/unary.h"

namespace op {

    expression::Expression sum(const expression::Expression& x) {
        return all_reduce(x, "reducers::sum");
    }
    expression::Expression sum(const expression::Expression& x, const std::vector<int>& axes) {
        return axis_reduce(x, "reducers::sum", axes);
    }
    expression::Expression prod(const expression::Expression& x) {
        return all_reduce(x, "reducers::product");
    }
    expression::Expression prod(const expression::Expression& x, const std::vector<int>& axes) {
        return axis_reduce(x, "reducers::product", axes);
    }
    expression::Expression max(const expression::Expression& x) {
        return all_reduce(x, "reducers::maximum");
    }
    expression::Expression max(const expression::Expression& x, const std::vector<int>& axes) {
        return axis_reduce(x, "reducers::maximum", axes);
    }
    expression::Expression min(const expression::Expression& x) {
        return all_reduce(x, "reducers::minimum");
    }
    expression::Expression min(const expression::Expression& x, const std::vector<int>& axes) {
        return axis_reduce(x, "reducers::minimum", axes);
    }
    expression::Expression mean(const expression::Expression& x) {
        auto sum_op = all_reduce(x, "reducers::sum");
        if (sum_op.dtype() == DTYPE_INT32) sum_op = astype(sum_op, DTYPE_DOUBLE);
        return op::eltdiv(sum_op, x.number_of_elements());
    }
    expression::Expression mean(const expression::Expression& x, const std::vector<int>& axes) {
        auto sum_op = axis_reduce(x, "reducers::sum", axes);
        if (sum_op.dtype() == DTYPE_INT32) sum_op = astype(sum_op, DTYPE_DOUBLE);
        return op::eltdiv(sum_op, x.number_of_elements() / sum_op.number_of_elements());
    }
    expression::Expression L2_norm(const expression::Expression& x) {
        auto sum_op = all_reduce(op::square(x), "reducers::sum");
        if (sum_op.dtype() == DTYPE_INT32) sum_op = astype(sum_op, DTYPE_DOUBLE);
        return op::sqrt(sum_op);
    }
    expression::Expression L2_norm(const expression::Expression& x, const std::vector<int>& axes) {
        auto sum_op = axis_reduce(op::square(x), "reducers::sum", axes);
        if (sum_op.dtype() == DTYPE_INT32) sum_op = astype(sum_op, DTYPE_DOUBLE);
        return op::sqrt(sum_op);
    }
    expression::Expression argmax(const expression::Expression& x) {
        return argument_all_reduce(x, "reducers::maximum");
    }
    expression::Expression argmax(const expression::Expression& x, const int& axis) {
        return argument_axis_reduce(x, "reducers::maximum", axis);
    }
    expression::Expression argmin(const expression::Expression& x) {
        return argument_all_reduce(x, "reducers::minimum");
    }
    expression::Expression argmin(const expression::Expression& x, const int& axis) {
        return argument_axis_reduce(x, "reducers::minimum", axis);
    }
}
