#ifndef DALI_ARRAY_OP_REDUCERS_H
#define DALI_ARRAY_OP_REDUCERS_H

#include "dali/array/array.h"
#include <string>

namespace op {
	// Compute the sum of all the elements in x
    Array sum(const Array& x);
    // Compute the sum of the elements in x along specific axes
    Array sum(const Array& x, const std::vector<int>& axes);
    // Compute the product of all the elements in x
    Array prod(const Array& x);
    // Compute the product of the elements in x along specific axes
    Array prod(const Array& x, const std::vector<int>& axes);
    // Find the max of all the elements in x
    Array max(const Array& x);
    // Find the max of the elements in x along specific axes
    Array max(const Array& x, const std::vector<int>& axes);
    // Find the min of all the elements in x
    Array min(const Array& x);
    // Find the min of the elements in x along specific axes
    Array min(const Array& x, const std::vector<int>& axes);
    // Mean
    // Return the sum of the elements in x divided by
    // the number of elements.
    // Note: If x is an int, then this returns the result as a double.
    // Otherwise the float-type remains unchanged.
    Array mean(const Array& x);
    Array L2_norm(const Array& x);
    // Array argmax(const Array& x);
    // Array argmin(const Array& x);

    Array mean(const Array& x, const std::vector<int>& axes);
    Array L2_norm(const Array& x, const std::vector<int>& axes);
    // Array argmax(const Array& x, const int& axis);
    // Array argmin(const Array& x, const int& axis);
} // namespace op

#endif
