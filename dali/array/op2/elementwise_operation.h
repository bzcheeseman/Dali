#ifndef DALI_ARRAY_OP2_ELEMENTWISE_OPERATION_H
#define DALI_ARRAY_OP2_ELEMENTWISE_OPERATION_H

#include "dali/array/dtype.h"
#include "dali/array/op2/operation.h"
#include <string>

namespace op {
    // elementwise kernel given by name. assumes
    // return type is unchanged from a's
    Operation elementwise(
        const Operation& a,
        const std::string& functor_name
    );

    // pair-wise kernel. Will type promote arguments
    // so that they have the same type when
    // given to the functor:
    // - float w/. double => double
    // - float w/. int => float
    // - double w/. int => double
    Operation elementwise(
        const Operation& a,
        const Operation& b,
        const std::string& functor_name
    );

    // call a kernel on a pair of arguments. Assumes
    // both arguments should be of the same type. Peforms
    // type promotion on the arguments if not. Will paste
    // and run the associated code `kernel_code` during
    // compilation and usage. (Warning: this might cause
    // collisions when a name is used multiple times)
    Operation binary_kernel_function(
        const Operation& a,
        const Operation& b,
        const std::string& function_name,
        const std::string& kernel_code
    );

    // Perform a type conversion by casting the values in x
    // to another dtype.
    Operation astype(const Operation& x, DType dtype);

    // type-promote arguments if necessary and check whether their
    // ranks are compatible (equal or one is a scalar)
    std::tuple<Operation, Operation> ensure_arguments_compatible(
        const Operation& a, const Operation& b
    );
} // namespace op2

#endif  // DALI_ARRAY_OP2_ELEMENTWISE_OPERATION_H
