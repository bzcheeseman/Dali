#ifndef DALI_ARRAY_OP2_REDUCERS_H
#define DALI_ARRAY_OP2_REDUCERS_H

#include <vector>
#include <string>

namespace expression {
    struct ExpressionGraph;
}  // namespace expression

namespace op {
	// Compute the sum of all the elements in x
    expression::ExpressionGraph sum(const expression::ExpressionGraph& x);
    // Compute the sum of the elements in x along specific axes
    expression::ExpressionGraph sum(const expression::ExpressionGraph& x, const std::vector<int>& axes);
    // Compute the product of all the elements in x
    expression::ExpressionGraph prod(const expression::ExpressionGraph& x);
    // Compute the product of the elements in x along specific axes
    expression::ExpressionGraph prod(const expression::ExpressionGraph& x, const std::vector<int>& axes);
    // Find the max of all the elements in x
    expression::ExpressionGraph max(const expression::ExpressionGraph& x);
    // Find the max of the elements in x along specific axes
    expression::ExpressionGraph max(const expression::ExpressionGraph& x, const std::vector<int>& axes);
    // Find the min of all the elements in x
    expression::ExpressionGraph min(const expression::ExpressionGraph& x);
    // Find the min of the elements in x along specific axes
    expression::ExpressionGraph min(const expression::ExpressionGraph& x, const std::vector<int>& axes);
    // Mean
    // Return the sum of the elements in x divided by
    // the number of elements.
    // Note: If x is an int, then this returns the result as a double.
    // Otherwise the float-type remains unchanged.
    expression::ExpressionGraph mean(const expression::ExpressionGraph& x);
    expression::ExpressionGraph L2_norm(const expression::ExpressionGraph& x);
    expression::ExpressionGraph argmax(const expression::ExpressionGraph& x);
    expression::ExpressionGraph argmin(const expression::ExpressionGraph& x);

    expression::ExpressionGraph mean(const expression::ExpressionGraph& x, const std::vector<int>& axes);
    expression::ExpressionGraph L2_norm(const expression::ExpressionGraph& x, const std::vector<int>& axes);
    expression::ExpressionGraph argmax(const expression::ExpressionGraph& x, const int& axis);
    expression::ExpressionGraph argmin(const expression::ExpressionGraph& x, const int& axis);
} // namespace op2

#endif
