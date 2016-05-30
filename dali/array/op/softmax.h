#ifndef DALI_ARRAY_OP_SOFTMAX_H
#define DALI_ARRAY_OP_SOFTMAX_H

class Array;
class AssignableArray;

namespace op {
	AssignableArray softmax(const Array& array, int axis, const double& temperature=1.0);
} // namespace op

#endif
