#ifndef DALI_ARRAY_OP2_REDUCERS_H
#define DALI_ARRAY_OP2_REDUCERS_H

#include <vector>
#include <string>

class Expression;

namespace op {
	// Compute the sum of all the elements in x
    Expression sum(const Expression& x);
    // Compute the sum of the elements in x along specific axes
    Expression sum(const Expression& x, const std::vector<int>& axes);
    // Compute the product of all the elements in x
    Expression prod(const Expression& x);
    // Compute the product of the elements in x along specific axes
    Expression prod(const Expression& x, const std::vector<int>& axes);
    // Find the max of all the elements in x
    Expression max(const Expression& x);
    // Find the max of the elements in x along specific axes
    Expression max(const Expression& x, const std::vector<int>& axes);
    // Find the min of all the elements in x
    Expression min(const Expression& x);
    // Find the min of the elements in x along specific axes
    Expression min(const Expression& x, const std::vector<int>& axes);
    // Mean
    // Return the sum of the elements in x divided by
    // the number of elements.
    // Note: If x is an int, then this returns the result as a double.
    // Otherwise the float-type remains unchanged.
    Expression mean(const Expression& x);
    Expression L2_norm(const Expression& x);
    Expression argmax(const Expression& x);
    Expression argmin(const Expression& x);

    Expression mean(const Expression& x, const std::vector<int>& axes);
    Expression L2_norm(const Expression& x, const std::vector<int>& axes);
    Expression argmax(const Expression& x, const int& axis);
    Expression argmin(const Expression& x, const int& axis);
} // namespace op2

#endif
