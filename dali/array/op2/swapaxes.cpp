#include "swapaxes.h"

#include "dali/array/op2/expression/expression.h"
#include "dali/array/op2/expression/array_wrapper.h"

namespace op {
    expression::Expression swapaxes(const expression::Expression& input, int axis1, int axis2) {
        int input_ndim = input.ndim();
        if (input_ndim == 0) return input;
        if (axis1 < 0) axis1 = input_ndim + axis1;
        if (axis2 < 0) axis2 = input_ndim + axis2;
        // no-op
        if (axis1 == axis2) return input;

        ASSERT2(0 <= axis1 && axis1 < input_ndim, utils::make_message("swapaxes"
            " axis1 (", axis1, ") must be less than ndim (", input_ndim, ")."));
        ASSERT2(0 <= axis2 && axis2 < input_ndim, utils::make_message("swapaxes"
            " axis2 (", axis2, ") must be less than ndim (", input_ndim, ")."));

        auto input_as_array = input.state_->as_array();
        if (input_as_array) {
            return expression::Expression(
                input_as_array->array_.swapaxes(axis1, axis2)
            );
        } else {
            ASSERT2(false, "swapaxes not yet implemented for non-Array expressions.");
        }
    }
}  // namespace
