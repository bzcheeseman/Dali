#ifndef DALI_ARRAY_OP2_CIRCULAR_CONVOLUTION_H
#define DALI_ARRAY_OP2_CIRCULAR_CONVOLUTION_H

struct Expression;

namespace op {
    Expression circular_convolution(const Expression& x, const Expression& weights);
}  // namespace op2

#endif  // DALI_ARRAY_OP2_CIRCULAR_CONVOLUTION_H
