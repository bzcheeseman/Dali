#ifndef DALI_ARRAY_OP_BINARY_H
#define DALI_ARRAY_OP_BINARY_H

#include <vector>

class Array;
template<typename OutType>
struct Assignable;

namespace op {
    Assignable<Array> add(const Array& left, const Array& right);
    Assignable<Array> add(const std::vector<Array>& arrays);
    Assignable<Array> sub(const Array& left, const Array& right);
    Assignable<Array> eltmul(const Array& left, const Array& right);
    Assignable<Array> eltdiv(const Array& left, const Array& right);
    Assignable<Array> pow(const Array& left, const Array& right);
    Assignable<Array> equals(const Array& left, const Array& right);
    Assignable<Array> circular_convolution(const Array& content, const Array& shift);
    Assignable<Array> prelu(const Array& x, const Array& weights);
}  // namespace op
#endif
