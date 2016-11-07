#include "transpose.h"

#include "dali/array/op2/expression/expression.h"
#include "dali/array/op2/expression/array_wrapper.h"

namespace op {
    expression::Expression transpose(const expression::Expression& input) {
        auto input_as_array = input.state_->as_array();
        if (input_as_array) {
            return expression::Expression(
                input_as_array->array_.transpose()
            );
        } else {
            ASSERT2(false, "transpose not yet implemented for non-Array expressions.");
        }
    }

    expression::Expression transpose(const expression::Expression& input, const std::vector<int>& axes) {
        return dimshuffle(input, axes);
    }

    expression::Expression dimshuffle(const expression::Expression& input, const std::vector<int>& axes) {
        int input_ndim = input.ndim();
        ASSERT2(axes.size() == input_ndim, utils::make_message("dimshuffle "
            "must receive as many axes as the dimensionality of the input "
            "(axes=", axes, ", ndim=", input_ndim,")."));

        auto input_as_array = input.state_->as_array();
        if (input_as_array) {
            return expression::Expression(
                input_as_array->array_.dimshuffle(axes)
            );
        } else {
            ASSERT2(false, "dimshuffle not yet implemented for non-Array expressions.");
        }
    }
}  // namespace op
