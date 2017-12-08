#include "binary.h"
#include "dali/array/op/reducers.h"
#include "dali/array/op/elementwise_operation.h"

namespace op {
    Array all_equals(Array left, Array right) {
        return op::prod(op::equals(left, right));
    }

    Array equals(Array left, Array right) {
        return op::elementwise(left, right, "functor::equals");
    }

    Array add(Array left, Array right) {
        throw std::runtime_error("add is not implemented.");
    }
    Array subtract(Array left, Array right) {
        throw std::runtime_error("subtract is not implemented.");
    }
    Array eltmul(Array left, Array right) {
        return op::elementwise(left, right, "functor::eltmul");
    }
    Array eltdiv(Array left, Array right) {
        return op::elementwise(left, right, "functor::eltdiv");
    }
}
