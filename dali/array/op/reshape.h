#ifndef DALI_ARRAY_OP_RESHAPE_H
#define DALI_ARRAY_OP_RESHAPE_H

#include <vector>

class Array;
class AssignableArray;

namespace op {
	// Join a sequence of arrays along an existing axis.
	AssignableArray concatenate(const std::vector<Array>& arrays, int axis);
	// Join a sequence of arrays along their last axis.
	AssignableArray hstack(const std::vector<Array>& arrays);
	// Stack arrays in sequence vertically (row wise).
	AssignableArray vstack(const std::vector<Array>& arrays);
} // namespace op

#endif
