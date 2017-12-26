#ifndef DALI_ARRAY_OP_BINARY_H
#define DALI_ARRAY_OP_BINARY_H

#include "dali/array/array.h"

namespace op {
    Array all_equals(Array left, Array right);
    Array all_close(Array left, Array right, Array atolerance);
    Array close(Array left, Array right, Array atolerance);
    Array add(Array left, Array right);
    Array subtract(Array left, Array right);
    Array eltmul(Array left, Array right);
    Array eltdiv(Array left, Array right);
    Array equals(Array left, Array right);

    Array pow(Array left, Array right);
    Array steep_sigmoid(Array x, Array aggressiveness);
    Array steep_sigmoid_backward(Array x, Array aggressiveness);
    Array clipped_relu(Array x, Array clipval);
    Array clipped_relu_backward(Array x, Array clipval);
    Array prelu(Array x, Array weights);
    Array prelu_backward_weights(Array a, Array grad);
    Array prelu_backward_inputs(Array a, Array weights);

    Array lessthanequal(Array a, Array b);
    Array greaterthanequal(Array a, Array b);
    Array eltmax(Array a, Array b);
    Array clip(Array a, Array b);
    Array eltmin(Array a, Array b);
    Array binary_cross_entropy(Array a, Array b);
    Array binary_cross_entropy_grad(Array a, Array b);
}

#endif
