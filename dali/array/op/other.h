#ifndef DALI_ARRAY_OP_OTHER_H
#define DALI_ARRAY_OP_OTHER_H

class Array;
template<typename OutType>
class Assignable;

namespace op {
    bool is_nan(const Array& x);
    Assignable<Array> all_equals(const Array& left, const Array& right);
    Assignable<Array> all_close(const Array& left, const Array& right, const double& atolerance);
    Assignable<Array> argsort(const Array& arr, int axis);
    Assignable<Array> argsort(const Array& arr);
}

#endif
