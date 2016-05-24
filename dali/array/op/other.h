#ifndef DALI_ARRAY_OP_OTHER_H
#define DALI_ARRAY_OP_OTHER_H

class Array;
class AssignableArray;

namespace op {
    bool is_nan(const Array& x);
    AssignableArray all_equals(const Array& left, const Array& right);
    AssignableArray all_close(const Array& left, const Array& right, const double& atolerance);
}

#endif
