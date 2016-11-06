#ifndef DALI_ARRAY_OP2_CIRCULAR_CONVOLUTION_H
#define DALI_ARRAY_OP2_CIRCULAR_CONVOLUTION_H

namespace expression {
    struct Expression;
}  // namespace expression

namespace op {
    expression::Expression circular_convolution(const expression::Expression& x, const expression::Expression& weights);
}  // namespace op2

#endif  // DALI_ARRAY_OP2_CIRCULAR_CONVOLUTION_H
