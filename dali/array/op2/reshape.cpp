#include "reshape.h"

#include "dali/array/op2/expression/expression.h"
#include "dali/array/op2/expression/array_wrapper.h"

namespace op {
    expression::Expression reshape(const expression::Expression& input, const std::vector<int>& newshape) {
        auto input_as_array = input.state_->as_array();
        if (input_as_array) {
            return expression::Expression(
                input_as_array->array_.reshape(newshape)
            );
        } else {
            auto input_rvalue = input.state_->as_rvalue();
            ASSERT2(input_rvalue, "reshape input must be an rvalue.");
            auto input_runnable = input_rvalue->as_runnable(input.preferred_device());
            auto input_runnable_dest_op_rvalue = input_runnable->destination_op()->as_rvalue();
            ASSERT2(input_runnable_dest_op_rvalue, "reshape input must be a runnable rvalue.");
            return reshape(
                expression::Expression(input_runnable_dest_op_rvalue),
                newshape
            );
        }
    }
}  // namespace
