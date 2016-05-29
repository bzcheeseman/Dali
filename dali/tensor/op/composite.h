#ifndef DALI_TENSOR_OP_COMPOSITE_H
#define DALI_TENSOR_OP_COMPOSITE_H

#include "dali/tensor/tensor.h"

namespace tensor_ops {
    Tensor dot_with_bias(const Tensor& inputs,
                         const Tensor& weight,
                         const Tensor& bias);


    Tensor multiple_dot_with_bias(const std::vector<Tensor>& inputs,
                                  const std::vector<Tensor>& weights,
                                  Tensor bias);

    /* Quadratic Form
     * ==============
     *
     * A composite operation of three tensor arguments left, middle, right.
     *
     * f(left, middle, right) = (left.T • middle) • right
     */
    Tensor quadratic_form(const Tensor& left, const Tensor& middle, const Tensor& right);
} // namespace tensor_ops

#endif
