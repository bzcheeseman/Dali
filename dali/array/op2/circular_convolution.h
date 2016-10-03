#ifndef DALI_ARRAY_OP2_CIRCULAR_CONVOLUTION_H
#define DALI_ARRAY_OP2_CIRCULAR_CONVOLUTION_H

struct Operation;

namespace op2 {
    Operation circular_convolution(const Operation& x, const Operation& weights);
}  // namespace op2

#endif  // DALI_ARRAY_OP2_CIRCULAR_CONVOLUTION_H
