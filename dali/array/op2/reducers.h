#ifndef DALI_ARRAY_OP2_REDUCERS_H
#define DALI_ARRAY_OP2_REDUCERS_H

class FusedOperation;

namespace op2 {
	// Compute the sum of all the elements in x
    FusedOperation sum(const FusedOperation& x);
    // Compute the product of all the elements in x
    FusedOperation prod(const FusedOperation& x);
    // Find the max of all the elements in x
    FusedOperation max(const FusedOperation& x);
    // Find the min of all the elements in x
    FusedOperation min(const FusedOperation& x);
    // Mean
    // Return the sum of the elements in x divided by
    // the number of elements.
    // Note: If x is an int, then this returns the result as a double.
    // Otherwise the float-type remains unchanged.
    FusedOperation mean(const FusedOperation& x);
    FusedOperation L2_norm(const FusedOperation& x);
} // namespace op2

#endif
