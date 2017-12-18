#ifndef DALI_ARRAY_OP2_ELEMENTWISE_OPERATION_H
#define DALI_ARRAY_OP2_ELEMENTWISE_OPERATION_H

#include "dali/array/array.h"
#include <string>

namespace op {
    // elementwise kernel given by name. assumes
    // return type is unchanged from a's
    Array elementwise(Array a, const std::string& functor_name);

    // pair-wise kernel. Will type promote arguments
    // so that they have the same type when
    // given to the functor:
    // - float w/. double => double
    // - float w/. int => float
    // - double w/. int => double
    Array elementwise(Array a, Array b, const std::string& functor_name);

    // call a kernel on a pair of arguments. Assumes
    // both arguments should be of the same type. Peforms
    // type promotion on the arguments if not. Will paste
    // and run the associated code `kernel_code` during
    // compilation and usage. (Warning: this might cause
    // collisions when a name is used multiple times)
    Array binary_kernel_function(
        Array a, Array b,
        const std::string& function_name,
        const std::string& kernel_code
    );

    // Perform a type conversion by casting the values in x
    // to another dtype. Use rounding when casting to integers
    // for more predictable results
    Array astype(Array x, DType dtype);
    // static_cast one type to another. This can cause unpredictable
    // behavior on floats->integers based on underlying
    // hardware/implementation
    Array unsafe_cast(Array x, DType dtype);
    // Perform rounding on a value to nearest integer.
    // Note: equivalent to floor(x + 0.5).
    Array round(Array x);

    // type-promote arguments if necessary and check whether their
    // ranks are compatible (equal or one is a scalar)
    std::tuple<Array, Array> ensure_arguments_compatible(
        const Array& a, const Array& b, const std::string&
    );
} // namespace op2

#endif  // DALI_ARRAY_OP2_ELEMENTWISE_OPERATION_H
