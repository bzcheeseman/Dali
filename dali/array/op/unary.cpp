#include "unary.h"
#include "dali/array/op/elementwise_operation.h"
#include "dali/array/jit/scalar_view.h"

namespace op {
    Array identity(Array x) {
        return elementwise(x, "functor::identity");
    }

    Array identity(float x) {
        return elementwise(jit::wrap_scalar(x), "functor::identity");
    }

    Array identity(double x) {
        return elementwise(jit::wrap_scalar(x), "functor::identity");
    }

    Array identity(int x) {
        return elementwise(jit::wrap_scalar(x), "functor::identity");
    }
}
