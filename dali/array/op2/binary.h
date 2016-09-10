#ifndef DALI_ARRAY_OP2_BINARY_H
#define DALI_ARRAY_OP2_BINARY_H

class Array;
template<typename OutType>
struct Assignable;

namespace op2 {
    Assignable<Array> add(const Array& a, const Array& b);
    Assignable<Array> sub(const Array& a, const Array& b);
    Assignable<Array> eltmul(const Array& left, const Array& right);
    Assignable<Array> eltdiv(const Array& left, const Array& right);
    Assignable<Array> pow(const Array& left, const Array& right);
    Assignable<Array> equals(const Array& left, const Array& right);
    Assignable<Array> prelu(const Array& x, const Array& weights);
}  // namespace op2

#endif  // DALI_ARRAY_OP2_BINARY_H
