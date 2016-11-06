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
            ASSERT2(false, "reshape not yet implemented for non-Array expressions.");
        }
    }
}  // namespace
