#include "assignment.h"
#include "dali/utils/make_message.h"
#include "dali/array/op/reducers.h"
#include "dali/array/expression/control_flow.h"

Assignment::Assignment(Array left, OPERATOR_T operator_t, Array right) :
        Expression(left.shape(),
                   left.dtype(),
                   {left, right},
                   left.offset(),
                   left.strides()), operator_t_(operator_t) {
}

const Array& Assignment::left() const {
    return arguments_[0];
}

const Array& Assignment::right() const {
    return arguments_[1];
}

Assignment::Assignment(const Assignment& other) :
        Assignment(other.left(), other.operator_t_, other.right()) {
}

expression_ptr Assignment::copy() const {
    return std::make_shared<Assignment>(*this);
}

expression_ptr Assignment::buffer_arg() const {
    return left().expression()->buffer_arg();
}

memory::Device Assignment::preferred_device() const {
    return left().preferred_device();
}

std::string Assignment::name() const {
    return utils::make_message(
        "Assignment[", operator_to_name(operator_t_), "]");
}

bool Assignment::is_axis_collapsible_with_axis_minus_one(int axis) const {
    return contiguous_memory();
}

bool Assignment::spans_entire_memory() const {
    return left().spans_entire_memory();
}

bool Assignment::is_assignable() const {
    return true;
}

expression_ptr Assignment::collapse_axis_with_axis_minus_one(int axis, const Array* owner) const {
    if (right().is_axis_collapsible_with_axis_minus_one(axis)) {
        return std::make_shared<Assignment>(
            left().collapse_axis_with_axis_minus_one(axis),
            operator_t_,
            right().collapse_axis_with_axis_minus_one(axis)
        );
    } else {
        auto collapsed_self = left().collapse_axis_with_axis_minus_one(axis);
        return op::control_dependency(Array(copy()), collapsed_self).expression();
    }
}


#define CONNECT_AUTO_ASSIGN(NAME)\
    if (owner != nullptr) {\
        return op::control_dependency(\
            *owner, (*owner).buffer_arg().NAME).expression();\
    } else {\
        Array assignment(copy());\
        return op::control_dependency(\
            assignment, assignment.buffer_arg().NAME).expression();\
    }\

expression_ptr Assignment::broadcast_to_shape(const std::vector<int>& new_shape, const Array* owner) const {
    CONNECT_AUTO_ASSIGN(broadcast_to_shape(new_shape))
}

expression_ptr Assignment::operator()(int idx, const Array* owner) const {
    CONNECT_AUTO_ASSIGN(operator() (idx))
}

expression_ptr Assignment::dimshuffle(const std::vector<int>& pattern, const Array* owner) const {
    CONNECT_AUTO_ASSIGN(dimshuffle(pattern))
}

expression_ptr Assignment::pluck_axis(int axis, const Slice& slice_unnormalized, const Array* owner) const {
    CONNECT_AUTO_ASSIGN(pluck_axis(axis, slice_unnormalized))
}

expression_ptr Assignment::_reshape(const std::vector<int>& new_shape, const Array* owner) const {
    CONNECT_AUTO_ASSIGN(reshape(new_shape))
}

expression_ptr Assignment::_squeeze(int axis, const Array* owner) const {
    CONNECT_AUTO_ASSIGN(squeeze(axis))
}

expression_ptr Assignment::_expand_dims(int new_axis, const Array* owner) const {
    CONNECT_AUTO_ASSIGN(expand_dims(new_axis))
}


namespace op {

Array autoreduce_assign(const Array& left, const Array& right) {
    Array assigned_right = right;
    Array assigned_left = left;
    if (!right.is_buffer()) {
        assigned_right = to_assignment(right);
    }
    if (!left.is_buffer() && !left.is_assignment()) {
        assigned_left = to_assignment(left);
    }
    ASSERT2(assigned_left.ndim() == right.ndim(), utils::make_message(
        "destination for autoreduce_assign must have "
        "the same dimensionality as the input (destination.ndim = ",
        assigned_left.ndim(), " vs input.ndim = ", right.ndim()));

    std::vector<int> reduction_axes;
    for (int i = 0; i < right.ndim(); i++) {
        if (assigned_left.shape()[i] != right.shape()[i]) {
            ASSERT2(assigned_left.shape()[i] == 1, utils::make_message(
                "Could not autoreduce_assign: destination must "
                "have all dimensions equal input dimensions or equal 1,"
                " but destination.shape[", i, "] = ", assigned_left.shape()[i],
                " vs input.shape[", i, "] = ", right.shape()[i]));
            reduction_axes.emplace_back(i);
        }
    }
    if (!reduction_axes.empty()) {
        assigned_right = op::sum(assigned_right, reduction_axes, /*keepdims=*/true);
    }
    return Array(std::make_shared<Assignment>(assigned_left, OPERATOR_T_ADD, assigned_right));
}

Array to_assignment(const Array& node) {
    return Array(std::make_shared<Assignment>(
                Array(node.shape(), node.dtype()),
                OPERATOR_T_EQL, Array(node.expression())));
}

Array assign(const Array& left, OPERATOR_T operator_t, const Array& right) {
    if (operator_t == OPERATOR_T_LSE) {
        return autoreduce_assign(left, right);
    }
    ASSERT2((right.ndim() == 0) | (right.ndim() == left.ndim()), utils::make_message(
            "Incompatible dimensions for assignment between left.ndim = ",
            left.ndim(), "(", left.full_expression_name(), ", and right.ndim = ", right.ndim(),
            " (", right.full_expression_name(), ")."));
    ASSERT2((right.ndim() == 0) | (right.shape() == left.shape()), utils::make_message(
            "Incompatible shapes for assignment between left.shape = ",
            left.shape(), "(", left.full_expression_name(), ", and right.shape = ", right.shape(),
            " (", right.full_expression_name(), ")."));
    if (operator_t == OPERATOR_T_EQL) {
        if (left.is_buffer() || left.spans_entire_memory()) {
            return Array(std::make_shared<Assignment>(left.buffer_arg(), operator_t, right));
        } else {
            return op::control_dependency(
                left,
                Array(std::make_shared<Assignment>(left.buffer_arg(), operator_t, right)));
        }
    } else {
        Array assigned_right = right;
        Array assigned_left = left;
        if (!right.expression()->supports_operator(operator_t)) {
            assigned_right = to_assignment(right);
        }
        if (!left.is_assignable()) {
            assigned_left = to_assignment(left);
        }
        if (assigned_left.is_assignment()) {
            auto left_assign = static_as_assignment(assigned_left);
            auto left_operator_t = left_assign->operator_t_;
            if (left_operator_t == OPERATOR_T_ADD) {
                assigned_right = assigned_right + left_assign->right();
            } else if (left_operator_t == OPERATOR_T_SUB) {
                assigned_right = assigned_right - left_assign->right();
            } else if (left_operator_t == OPERATOR_T_MUL) {
                assigned_right = assigned_right * left_assign->right();
            } else if (left_operator_t == OPERATOR_T_EQL) {
                // TODO(jonathan): factorize mapping from operator -> binary func
                // into one place
                if (operator_t == OPERATOR_T_ADD) {
                    assigned_right = left_assign->right() + assigned_right;
                    operator_t = OPERATOR_T_EQL;
                } else if (operator_t == OPERATOR_T_SUB) {
                    assigned_right = left_assign->right() - assigned_right;
                    operator_t = OPERATOR_T_EQL;
                } else if (operator_t == OPERATOR_T_DIV) {
                    assigned_right = left_assign->right() / assigned_right;
                    operator_t = OPERATOR_T_EQL;
                } else if (operator_t == OPERATOR_T_MUL) {
                    assigned_right = left_assign->right() * assigned_right;
                    operator_t = OPERATOR_T_EQL;
                }
            } else {
                // TODO(jonathan): fill this in
                ASSERT2(false, "not sure what to do");
            }
            assigned_left = left_assign->left();
        }
        // a temp is added so that non overwriting operators
        // can be run independently from the right side's evaluation.
        return Array(std::make_shared<Assignment>(assigned_left, operator_t, assigned_right));
    }
}

Assignment* static_as_assignment(const Array& arr) {
    return static_cast<Assignment*>(arr.expression().get());
}

}  // namespace op
