#ifndef DALI_ARRAY_OP_UNARY_H
#define DALI_ARRAY_OP_UNARY_H

class Array;
class AssignableArray;

namespace op {
    // Assign one array to another piece of memory as-is
    // this also has the side-effect of a strided memory view
    // non-strided in the output (as it is no longer a view)
    //
    // if always_copy is true:  the input array x is always
    //           element-wise copied to the destination
    // if always_copy is false: input array is element-wise copied to
    //           destination except if the output and
    //           x are the same array.
    AssignableArray identity(const Array& x, const bool& always_copy=true);
    AssignableArray sigmoid(const Array& x);
    AssignableArray tanh(const Array& x);
    AssignableArray relu(const Array& x);
    AssignableArray eltinv(const Array& x);
    AssignableArray exp(const Array& x);
    AssignableArray log(const Array& x);
    AssignableArray log_or_zero(const Array& x);
    AssignableArray abs(const Array& x);
    AssignableArray sign(const Array& x);
    AssignableArray square(const Array& x);
    AssignableArray softplus(const Array& x);
    AssignableArray cube(const Array& x);
    AssignableArray sqrt(const Array& x);
    AssignableArray rsqrt(const Array& x);
} // namespace op

#endif
