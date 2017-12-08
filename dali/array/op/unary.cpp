#include "unary.h"
#include "dali/array/op/elementwise_operation.h"
#include "dali/array/jit/scalar_view.h"

namespace op {
    Array identity(Array x) {
        return elementwise(x, "functor::identity");
    }

    Array identity(float x) {
        return identity(jit::wrap_scalar(x));
    }

    Array identity(double x) {
        return identity(jit::wrap_scalar(x));
    }

    Array identity(int x) {
        return identity(jit::wrap_scalar(x));
    }

    Array sqrt(Array x) {
        return elementwise(x, "functor::sqrt");
    }

    Array square(Array x) {
        return elementwise(x, "functor::square");
    }
}
