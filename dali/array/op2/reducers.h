#ifndef DALI_ARRAY_OP2_REDUCERS_H
#define DALI_ARRAY_OP2_REDUCERS_H

#include <vector>
#include <string>

namespace expression {
    struct Expression;
}  // namespace expression

namespace op {
	// Compute the sum of all the elements in x
    expression::Expression sum(const expression::Expression& x);
    // Compute the sum of the elements in x along specific axes
    expression::Expression sum(const expression::Expression& x, const std::vector<int>& axes);
    // Compute the product of all the elements in x
    expression::Expression prod(const expression::Expression& x);
    // Compute the product of the elements in x along specific axes
    expression::Expression prod(const expression::Expression& x, const std::vector<int>& axes);
    // Find the max of all the elements in x
    expression::Expression max(const expression::Expression& x);
    // Find the max of the elements in x along specific axes
    expression::Expression max(const expression::Expression& x, const std::vector<int>& axes);
    // Find the min of all the elements in x
    expression::Expression min(const expression::Expression& x);
    // Find the min of the elements in x along specific axes
    expression::Expression min(const expression::Expression& x, const std::vector<int>& axes);
    // Mean
    // Return the sum of the elements in x divided by
    // the number of elements.
    // Note: If x is an int, then this returns the result as a double.
    // Otherwise the float-type remains unchanged.
    expression::Expression mean(const expression::Expression& x);
    expression::Expression L2_norm(const expression::Expression& x);
    expression::Expression argmax(const expression::Expression& x);
    expression::Expression argmin(const expression::Expression& x);

    expression::Expression mean(const expression::Expression& x, const std::vector<int>& axes);
    expression::Expression L2_norm(const expression::Expression& x, const std::vector<int>& axes);
    expression::Expression argmax(const expression::Expression& x, const int& axis);
    expression::Expression argmin(const expression::Expression& x, const int& axis);
} // namespace op2

#endif
