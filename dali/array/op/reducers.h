#ifndef DALI_ARRAY_OP_REDUCERS_H
#define DALI_ARRAY_OP_REDUCERS_H

class Array;
template<typename OutType>
class Assignable;

namespace op {
    Assignable<Array> sum(const Array& x);
    Assignable<Array> L2_norm(const Array& x);
    Assignable<Array> L2_norm(const Array& x, const int& axis);
    Assignable<Array> mean(const Array& x);
    Assignable<Array> min(const Array& x);
    Assignable<Array> max(const Array& x);
    Assignable<Array> sum(const Array& x, const int& axis);
    Assignable<Array> mean(const Array& x, const int& axis);
    Assignable<Array> min(const Array& x, const int& axis);
    Assignable<Array> max(const Array& x, const int& axis);
    Assignable<Array> argmin(const Array& x, const int& axis);
    Assignable<Array> argmax(const Array& x, const int& axis);
}; // namespace op
#endif
