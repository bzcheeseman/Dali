#include "transpose.h"

#include "dali/utils/make_message.h"

namespace op {
    expression::ExpressionGraph transpose(const expression::ExpressionGraph& input) {
        auto input_as_array = input.state_->as_array();
        if (input_as_array) {
            return expression::ExpressionGraph(
                input_as_array->array_.transpose()
            );
        } else {
            ASSERT2(false, "transpose not yet implemented for non-Array expressions.");
        }
    }

    expression::ExpressionGraph transpose(const expression::ExpressionGraph& input, const std::vector<int>& axes) {
        return dimshuffle(input, axes);
    }

    expression::ExpressionGraph dimshuffle(const expression::ExpressionGraph& input, const std::vector<int>& axes) {
        int input_ndim = input.ndim();
        ASSERT2(axes.size() == input_ndim, utils::make_message("dimshuffle "
            "must receive as many axes as the dimensionality of the input "
            "(axes=", axes, ", ndim=", input_ndim,")."));

        auto input_as_array = input.state_->as_array();
        if (input_as_array) {
            return expression::ExpressionGraph(
                input_as_array->array_.dimshuffle(axes)
            );
        } else {
            ASSERT2(false, "dimshuffle not yet implemented for non-Array expressions.");
        }
    }
}  // namespace op
