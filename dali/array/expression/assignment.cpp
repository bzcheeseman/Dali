#include "assignment.h"
#include "dali/utils/make_message.h"
#include "dali/array/op/reducers.h"
#include "dali/array/expression/control_flow.h"

// TODO should pass strides + offset to Expression
Assignment::Assignment(Array left, OPERATOR_T operator_t, Array right) :
        Expression(left.shape(),
                   left.dtype()),
                   left_(left), operator_t_(operator_t), right_(right) {

}

Assignment::Assignment(const Assignment& other) :
        Assignment(other.left_, other.operator_t_, other.right_) {
}

expression_ptr Assignment::copy() const {
    return std::make_shared<Assignment>(*this);
}

memory::Device Assignment::preferred_device() const {
    return left_.preferred_device();
}

std::vector<Array> Assignment::arguments() const {
    return {left_, right_};
}

std::string Assignment::name() const {
    return utils::make_message(
        "Assignment[", operator_to_name(operator_t_), ", ",
        shape_, strides_, "]");
}

bool Assignment::is_axis_collapsible_with_axis_minus_one(int axis) const {
    return contiguous_memory();
}

expression_ptr Assignment::collapse_axis_with_axis_minus_one(int axis) const {
    if (right_.is_axis_collapsible_with_axis_minus_one(axis)) {
        return std::make_shared<Assignment>(
            left_.collapse_axis_with_axis_minus_one(axis),
            operator_t_,
            right_.collapse_axis_with_axis_minus_one(axis)
        );
    } else {
        auto collapsed_self = left_.collapse_axis_with_axis_minus_one(axis);
        return std::make_shared<ControlFlow>(
            collapsed_self, std::vector<Array>({
                Array(copy())
            })
        );
    }
}

namespace op {

Array autoreduce_assign(const Array& left, const Array& right) {
    Array assigned_right = right;
    if (!right.is_buffer()) {
        assigned_right = to_assignment(right);
    }
    ASSERT2(left.ndim() == right.ndim(), utils::make_message(
        "destination for autoreduce_assign must have "
        "the same dimensionality as the input (destination.ndim = ",
        left.ndim(), " vs input.ndim = ", right.ndim()));

    std::vector<int> reduction_axes;
    for (int i = 0; i < right.ndim(); i++) {
        if (left.shape()[i] != right.shape()[i]) {
            ASSERT2(left.shape()[i] == 1, utils::make_message(
                "Could not autoreduce_assign: destination must "
                "have all dimensions equal input dimensions or equal 1,"
                " but destination.shape[", i, "] = ", left.shape()[i],
                " vs input.shape[", i, "] = ", right.shape()[i]));
            reduction_axes.emplace_back(i);
        }
    }
    if (!reduction_axes.empty()) {
        assigned_right = op::sum(assigned_right, reduction_axes);
    }
    return Array(std::make_shared<Assignment>(left, OPERATOR_T_ADD, assigned_right));
}

Array to_assignment(const Array& node) {
    return assign(Array::zeros(node.shape(), node.dtype()),
                  OPERATOR_T_EQL,
                  Array(node.expression()));
}

Array assign(const Array& left, OPERATOR_T operator_t, const Array& right) {
    if (operator_t == OPERATOR_T_EQL) {
        return Array(std::make_shared<Assignment>(left, operator_t, right));
    } else if (operator_t == OPERATOR_T_LSE) {
        return autoreduce_assign(left, right);
    } else {
        Array assigned_right = right;
        if (!right.is_buffer()) {
            assigned_right = to_assignment(right);
        }
        // a temp is added so that non overwriting operators
        // can be run independently from the right side's evaluation.
        return Array(std::make_shared<Assignment>(left, operator_t, assigned_right));
    }
}

Assignment* static_as_assignment(const Array& arr) {
    return static_cast<Assignment*>(arr.expression().get());
}

}  // namespace op
