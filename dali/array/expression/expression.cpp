#include "expression.h"

Expression::Expression(const std::vector<int>& shape,
           DType dtype,
           int offset,
           const std::vector<int>& strides) :
        shape_(shape),
        dtype_(dtype),
        offset_(offset),
        strides_(strides) {
}
