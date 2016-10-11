#ifndef DALI_ARRAY_OP_UNARY_H
#define DALI_ARRAY_OP_UNARY_H

class Array;
template<typename OutType>
struct Assignable;

namespace old_op {
    // Assign one array to another piece of memory as-is
    // this also has the side-effect of a strided memory view
    // non-strided in the output (as it is no longer a view)
    //
    // if always_copy is true:  the input array x is always
    //           element-wise copied to the destination
    // if always_copy is false: input array is element-wise copied to
    //           destination except if the output and
    //           x are the same array.
    Assignable<Array> identity(const Array& x, const bool& always_copy=true);
    Assignable<Array> sigmoid(const Array& x);
    Assignable<Array> tanh(const Array& x);
    Assignable<Array> relu(const Array& x);
    Assignable<Array> eltinv(const Array& x);
    Assignable<Array> exp(const Array& x);
    Assignable<Array> log(const Array& x);
    Assignable<Array> log_or_zero(const Array& x);
    Assignable<Array> abs(const Array& x);
    Assignable<Array> sign(const Array& x);
    Assignable<Array> square(const Array& x);
    Assignable<Array> softplus(const Array& x);
    Assignable<Array> cube(const Array& x);
    Assignable<Array> sqrt(const Array& x);
    Assignable<Array> rsqrt(const Array& x);
} // namespace op

#endif
