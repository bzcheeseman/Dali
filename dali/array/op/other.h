#ifndef DALI_ARRAY_OP_OTHER_H
#define DALI_ARRAY_OP_OTHER_H

class Array;
template<typename OutType>
class Assignable;

namespace op {
    Assignable<Array> any_isnan(const Array& array);
    Assignable<Array> any_isinf(const Array& array);
    Assignable<Array> any_isnan(const Array& array, int axis);
    Assignable<Array> any_isinf(const Array& array, int axis);
    Assignable<Array> all_equals(const Array& left, const Array& right);
    Assignable<Array> all_close(const Array& left, const Array& right, const double& atolerance);
    Assignable<Array> argsort(const Array& array, int axis);
    Assignable<Array> argsort(const Array& array);
}

#endif
