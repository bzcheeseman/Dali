#include "control_flow.h"

std::vector<Array> join_array(std::vector<Array> left, const std::vector<Array>& right) {
    left.insert(left.end(), right.begin(), right.end());
    return left;
}

const Array& ControlFlow::left() const {
    return arguments_[0];
}

// TODO should pass strides + offset to Expression
ControlFlow::ControlFlow(Array left, const std::vector<Array>& conditions) :
        Expression(left.shape(),
                   left.dtype(),
                   join_array({left}, conditions),
                   left.offset(),
                   left.strides()) {
}

ControlFlow::ControlFlow(const ControlFlow& other) :
        ControlFlow(other.left(), std::vector<Array>(other.arguments_.begin() + 1, other.arguments_.end())) {
}

expression_ptr ControlFlow::copy() const {
    return std::make_shared<ControlFlow>(*this);
}

memory::Device ControlFlow::preferred_device() const {
    return left().preferred_device();
}

bool ControlFlow::is_axis_collapsible_with_axis_minus_one(int axis) const {
    return left().is_axis_collapsible_with_axis_minus_one(axis);
}

bool ControlFlow::spans_entire_memory() const {
    return left().spans_entire_memory();
}

bool ControlFlow::supports_operator(OPERATOR_T operator_t) const {
    return left().expression()->supports_operator(operator_t);
}

bool ControlFlow::is_assignable() const {
    return left().is_assignable();
}

bool ControlFlow::all_conditions_are_met() const {
    return std::all_of(arguments_.begin() + 1, arguments_.end(), [](const Array& array) {
        return array.is_buffer();
    });
}

std::vector<Array> ControlFlow::conditions() const {
    return std::vector<Array>(arguments_.begin() + 1, arguments_.end());
}

expression_ptr ControlFlow::buffer_arg() const {
    return left().expression()->buffer_arg();
}

namespace op {
ControlFlow* static_as_control_flow(const Array& arr) {
    return static_cast<ControlFlow*>(arr.expression().get());
}

Array control_dependency(Array condition, Array result) {
    if (condition.is_control_flow()) {
        ControlFlow* cflow = static_as_control_flow(condition);
        if (cflow->all_conditions_are_met()) {
            return result;
        }
        if (cflow->left().is_buffer()) {
            return Array(std::make_shared<ControlFlow>(result, cflow->conditions()));
        }
    }
    return Array(std::make_shared<ControlFlow>(result, std::vector<Array>({condition})));
}

}  // namespace op
