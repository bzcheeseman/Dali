#ifndef DALI_ARRAY_OP2_REDUCERS_H
#define DALI_ARRAY_OP2_REDUCERS_H

#include <vector>

class FusedOperation;

namespace op2 {
	// Compute the sum of all the elements in x
    FusedOperation sum(const FusedOperation& x);
    // Compute the sum of the elements in x along specific axes
    FusedOperation sum(const FusedOperation& x, const std::vector<int>& axes);
    // Compute the product of all the elements in x
    FusedOperation prod(const FusedOperation& x);
    // Compute the product of the elements in x along specific axes
    FusedOperation prod(const FusedOperation& x, const std::vector<int>& axes);
    // Find the max of all the elements in x
    FusedOperation max(const FusedOperation& x);
    // Find the max of the elements in x along specific axes
    FusedOperation max(const FusedOperation& x, const std::vector<int>& axes);
    // Find the min of all the elements in x
    FusedOperation min(const FusedOperation& x);
    // Find the min of the elements in x along specific axes
    FusedOperation min(const FusedOperation& x, const std::vector<int>& axes);
    // Mean
    // Return the sum of the elements in x divided by
    // the number of elements.
    // Note: If x is an int, then this returns the result as a double.
    // Otherwise the float-type remains unchanged.
    FusedOperation mean(const FusedOperation& x);
    FusedOperation L2_norm(const FusedOperation& x);
    FusedOperation argmax(const FusedOperation& x);
    FusedOperation argmin(const FusedOperation& x);

    FusedOperation mean(const FusedOperation& x, const std::vector<int>& axes);
    FusedOperation L2_norm(const FusedOperation& x, const std::vector<int>& axes);
    FusedOperation argmax(const FusedOperation& x, const int& axis);
    FusedOperation argmin(const FusedOperation& x, const int& axis);
} // namespace op2

#endif
