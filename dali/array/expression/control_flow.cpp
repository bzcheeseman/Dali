#include "control_flow.h"

#include <algorithm>

#include "dali/array/expression/assignment.h"

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

#define CONNECT_AUTO_ASSIGN(NAME)\
    if (owner != nullptr) {\
        return op::control_dependency(*owner, (*owner).buffer_arg().NAME).expression();\
    } else {\
        Array cflow(copy());\
        return op::control_dependency(cflow, cflow.buffer_arg().NAME).expression();\
    }\

// #define CONNECT_AUTO_ASSIGN(NAME)\
//     Array assignment = op::to_assignment(copy());\
//     if (owner != nullptr) {\
//         owner->set_expression(assignment.expression());\
//         return op::control_dependency(\
//             *owner, (*owner).buffer_arg().NAME).expression();\
//     }\
//     return op::control_dependency(\
//         assignment, assignment.buffer_arg().NAME).expression();\

expression_ptr ControlFlow::broadcast_to_shape(const std::vector<int>& new_shape, const Array* owner) const {
    CONNECT_AUTO_ASSIGN(broadcast_to_shape(new_shape))
}

expression_ptr ControlFlow::operator()(int idx, const Array* owner) const {
    CONNECT_AUTO_ASSIGN(operator() (idx))
}

expression_ptr ControlFlow::dimshuffle(const std::vector<int>& pattern, const Array* owner) const {
    CONNECT_AUTO_ASSIGN(dimshuffle(pattern))
}

expression_ptr ControlFlow::_reshape(const std::vector<int>& new_shape, const Array* owner) const {
    CONNECT_AUTO_ASSIGN(reshape(new_shape))
}

expression_ptr ControlFlow::pluck_axis(int axis, const Slice& slice_unnormalized, const Array* owner) const {
    CONNECT_AUTO_ASSIGN(pluck_axis(axis, slice_unnormalized))
}

expression_ptr ControlFlow::squeeze(int axis, const Array* owner) const {
    CONNECT_AUTO_ASSIGN(squeeze(axis))
}

expression_ptr ControlFlow::expand_dims(int new_axis, const Array* owner) const {
    CONNECT_AUTO_ASSIGN(expand_dims(new_axis))
}

expression_ptr ControlFlow::broadcast_axis(int axis, const Array* owner) const {
    CONNECT_AUTO_ASSIGN(broadcast_axis(axis))
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
