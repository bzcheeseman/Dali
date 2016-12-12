#ifndef DALI_ARRAY_EXPRESSION_EXPRESSION_H
#define DALI_ARRAY_EXPRESSION_EXPRESSION_H

#include <vector>

#include "dali/array/dtype.h"
#include "dali/array/memory/device.h"


struct Expression  {
    std::vector<int> shape_;
    DType            dtype_;
    int              offset_; // expressing in number of numbers (not bytes)
    std::vector<int> strides_;

    virtual memory::Device preferred_device() const = 0;

    Expression(const std::vector<int>& shape,
               DType dtype,
               int offset=0,
               const std::vector<int>& strides={});
};

#endif  // DALI_ARRAY_EXPRESSION_EXPRESSION_H
