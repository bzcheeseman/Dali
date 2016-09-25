#include "reducers.h"
#include "dali/array/op2/fused_operation.h"
#include "dali/array/op2/binary.h"
#include "dali/array/op2/unary.h"

namespace op2 {
    FusedOperation sum(const FusedOperation& x) {
        return all_reduce(x, "reducers::sum");
    }
    FusedOperation sum(const FusedOperation& x, const std::vector<int>& axes) {
        return axis_reduce(x, "reducers::sum", axes);
    }
    FusedOperation prod(const FusedOperation& x) {
        return all_reduce(x, "reducers::product");
    }
    FusedOperation prod(const FusedOperation& x, const std::vector<int>& axes) {
        return axis_reduce(x, "reducers::product", axes);
    }
    FusedOperation max(const FusedOperation& x) {
        return all_reduce(x, "reducers::maximum");
    }
    FusedOperation max(const FusedOperation& x, const std::vector<int>& axes) {
        return axis_reduce(x, "reducers::maximum", axes);
    }
    FusedOperation min(const FusedOperation& x) {
        return all_reduce(x, "reducers::minimum");
    }
    FusedOperation min(const FusedOperation& x, const std::vector<int>& axes) {
        return axis_reduce(x, "reducers::minimum", axes);
    }
    FusedOperation mean(const FusedOperation& x) {
        auto sum_op = all_reduce(x, "reducers::sum");
        if (sum_op.dtype() == DTYPE_INT32) sum_op = astype(sum_op, DTYPE_DOUBLE);
        return op2::eltdiv(sum_op, x.number_of_elements());
    }
    FusedOperation mean(const FusedOperation& x, const std::vector<int>& axes) {
        auto sum_op = axis_reduce(x, "reducers::sum", axes);
        if (sum_op.dtype() == DTYPE_INT32) sum_op = astype(sum_op, DTYPE_DOUBLE);
        return op2::eltdiv(sum_op, x.number_of_elements() / sum_op.number_of_elements());
    }
    FusedOperation L2_norm(const FusedOperation& x) {
        auto sum_op = all_reduce(op2::square(x), "reducers::sum");
        if (sum_op.dtype() == DTYPE_INT32) sum_op = astype(sum_op, DTYPE_DOUBLE);
        return op2::sqrt(sum_op);
    }
    FusedOperation L2_norm(const FusedOperation& x, const std::vector<int>& axes) {
        auto sum_op = axis_reduce(op2::square(x), "reducers::sum", axes);
        if (sum_op.dtype() == DTYPE_INT32) sum_op = astype(sum_op, DTYPE_DOUBLE);
        return op2::sqrt(sum_op);
    }
    FusedOperation argmax(const FusedOperation& x) {
        return argument_all_reduce(x, "reducers::maximum");
    }
    FusedOperation argmax(const FusedOperation& x, const int& axis) {
        return argument_axis_reduce(x, "reducers::maximum", axis);
    }
    FusedOperation argmin(const FusedOperation& x) {
        return argument_all_reduce(x, "reducers::minimum");
    }
    FusedOperation argmin(const FusedOperation& x, const int& axis) {
        return argument_axis_reduce(x, "reducers::minimum", axis);
    }
}
