#ifndef DALI_ARRAY_OP2_REDUCERS_H
#define DALI_ARRAY_OP2_REDUCERS_H

#include <vector>
#include <string>

class Operation;

namespace op2 {
	// Compute the sum of all the elements in x
    Operation sum(const Operation& x);
    // Compute the sum of the elements in x along specific axes
    Operation sum(const Operation& x, const std::vector<int>& axes);
    // Compute the product of all the elements in x
    Operation prod(const Operation& x);
    // Compute the product of the elements in x along specific axes
    Operation prod(const Operation& x, const std::vector<int>& axes);
    // Find the max of all the elements in x
    Operation max(const Operation& x);
    // Find the max of the elements in x along specific axes
    Operation max(const Operation& x, const std::vector<int>& axes);
    // Find the min of all the elements in x
    Operation min(const Operation& x);
    // Find the min of the elements in x along specific axes
    Operation min(const Operation& x, const std::vector<int>& axes);
    // Mean
    // Return the sum of the elements in x divided by
    // the number of elements.
    // Note: If x is an int, then this returns the result as a double.
    // Otherwise the float-type remains unchanged.
    Operation mean(const Operation& x);
    Operation L2_norm(const Operation& x);
    Operation argmax(const Operation& x);
    Operation argmin(const Operation& x);

    Operation mean(const Operation& x, const std::vector<int>& axes);
    Operation L2_norm(const Operation& x, const std::vector<int>& axes);
    Operation argmax(const Operation& x, const int& axis);
    Operation argmin(const Operation& x, const int& axis);
} // namespace op2

#endif
