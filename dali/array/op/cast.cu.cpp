#include "cast.h"
#include "dali/array/array.h"
#include "dali/array/lazy/cast.h"
#include "dali/array/op/unary.h"

namespace op {
    Assignable<Array> astype(const Array& a, DType dtype) {
        if (a.dtype() == dtype) {
            return op::identity(a, false);
        }
        if (dtype == DTYPE_FLOAT) {
            return lazy::astype<float>(a);
        } else if (dtype == DTYPE_DOUBLE) {
            return lazy::astype<double>(a);
        } else if (dtype == DTYPE_INT32) {
            return lazy::astype<int>(a);
        } else {
            ASSERT2(false, utils::MS() << "astype argument dtype must be one of " DALI_ACCEPTABLE_DTYPE_STR ".");
            return op::identity(a, false);
        }
    }
}
