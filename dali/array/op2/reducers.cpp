#include "reducers.h"
#include "dali/array/op2/fused_operation.h"
#include "dali/array/op2/binary.h"
#include "dali/array/op2/unary.h"

namespace op2 {
    FusedOperation sum(const FusedOperation& x) {
        return all_reduce(x, "reducers::sum");
    }
    FusedOperation prod(const FusedOperation& x) {
        return all_reduce(x, "reducers::product");
    }
    FusedOperation max(const FusedOperation& x) {
        return all_reduce(x, "reducers::maximum");
    }
    FusedOperation min(const FusedOperation& x) {
        return all_reduce(x, "reducers::minimum");
    }
    FusedOperation mean(const FusedOperation& x) {
        auto sum_op = all_reduce(x, "reducers::sum");
        if (sum_op.dtype() == DTYPE_INT32) sum_op = astype(sum_op, DTYPE_DOUBLE);
        return op2::eltdiv(sum_op, x.number_of_elements());
    }
    FusedOperation L2_norm(const FusedOperation& x) {
        auto sum_op = all_reduce(op2::square(x), "reducers::sum");
        if (sum_op.dtype() == DTYPE_INT32) sum_op = astype(sum_op, DTYPE_DOUBLE);
        return op2::sqrt(sum_op);
    }
}
