#include "control_flow.h"
// TODO should pass strides + offset to Expression
ControlFlow::ControlFlow(Array left, const std::vector<Array>& conditions) :
        Expression(left.shape(),
                   left.dtype(),
                   left.offset(),
                   left.strides()),
                   left_(left), conditions_(conditions) {
}

ControlFlow::ControlFlow(const ControlFlow& other) :
        ControlFlow(other.left_, other.conditions_) {
}

expression_ptr ControlFlow::copy() const {
    return std::make_shared<ControlFlow>(*this);
}

memory::Device ControlFlow::preferred_device() const {
    return left_.preferred_device();
}

std::vector<Array> ControlFlow::arguments() const {
    std::vector<Array> args({left_,});
    args.insert(args.end(), conditions_.begin(), conditions_.end());
    return args;
}

bool ControlFlow::is_axis_collapsible_with_axis_minus_one(int axis) const {
    return left_.is_axis_collapsible_with_axis_minus_one(axis);
}

bool ControlFlow::spans_entire_memory() const {
    return left_.spans_entire_memory();
}

bool ControlFlow::supports_operator(OPERATOR_T operator_t) const {
    return left_.expression()->supports_operator(operator_t);
}

bool ControlFlow::is_assignable() const {
    return left_.is_assignable();
}

namespace op {
ControlFlow* static_as_control_flow(const Array& arr) {
    return static_cast<ControlFlow*>(arr.expression().get());
}

Array control_dependency(Array condition, Array result) {
    if (condition.is_control_flow()) {
        ControlFlow* cflow = static_as_control_flow(condition);
        if (cflow->left_.is_buffer()) {
            return Array(std::make_shared<ControlFlow>(result, cflow->conditions_));
        }
    }
    return Array(std::make_shared<ControlFlow>(result, std::vector<Array>({condition})));
}

}  // namespace op
