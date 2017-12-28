#include "assignment.h"
#include "dali/utils/make_message.h"
#include "dali/array/op/reducers.h"
#include "dali/array/expression/control_flow.h"
#include "dali/array/op/binary.h"

Assignment::Assignment(Array left, OPERATOR_T operator_t, Array right) :
        Expression(left.shape(),
                   left.dtype(),
                   left.offset(),
                   left.strides()),
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
        "Assignment[", operator_to_name(operator_t_), "]");
}

bool Assignment::is_axis_collapsible_with_axis_minus_one(int axis) const {
    return contiguous_memory();
}

bool Assignment::spans_entire_memory() const {
    return left_.spans_entire_memory();
}

expression_ptr Assignment::collapse_axis_with_axis_minus_one(int axis) const {
    if (right_.is_axis_collapsible_with_axis_minus_one(axis)) {
        return std::make_shared<Assignment>(
            left_.collapse_axis_with_axis_minus_one(axis),
            operator_t_,
            right_.collapse_axis_with_axis_minus_one(axis)
        );
    } else {
        auto collapsed_self = left_.collapse_axis_with_axis_minus_one(axis);
        return std::make_shared<ControlFlow>(
            collapsed_self, std::vector<Array>({
                Array(copy())
            })
        );
    }
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
    return assign(Array::zeros(node.shape(), node.dtype()),
                  OPERATOR_T_EQL,
                  Array(node.expression()));
}

Array assign(const Array& left, OPERATOR_T operator_t, const Array& right) {
    if (operator_t == OPERATOR_T_EQL) {
        if (left.is_buffer() || left.spans_entire_memory()) {
            return Array(std::make_shared<Assignment>(left.buffer_arg(), operator_t, right));
        } else {
            return Array(std::make_shared<ControlFlow>(
                Array(std::make_shared<Assignment>(left.buffer_arg(), operator_t, right)),
                std::vector<Array>({left})
            ));
        }
    } else if (operator_t == OPERATOR_T_LSE) {
        return autoreduce_assign(left, right);
    } else {
        Array assigned_right = right;
        Array assigned_left = left;
        if (!right.expression()->supports_operator(operator_t)) {
            assigned_right = to_assignment(right);
        }
        if (!left.is_buffer() && !left.is_assignment()) {
            assigned_left = to_assignment(left);
        }
        if (assigned_left.is_assignment()) {
            auto left_assign = static_as_assignment(assigned_left);
            auto left_operator_t = left_assign->operator_t_;
            if (left_operator_t == OPERATOR_T_ADD) {
                assigned_right = op::add(assigned_right, left_assign->right_);
            } else if (left_operator_t == OPERATOR_T_SUB) {
                assigned_right = op::subtract(assigned_right, left_assign->right_);
            } else if (left_operator_t == OPERATOR_T_SUB) {
                assigned_right = op::subtract(assigned_right, left_assign->right_);
            } else if (left_operator_t == OPERATOR_T_EQL) {
                // TODO(jonathan): factorize mapping from operator -> binary func
                // into one place
                if (operator_t == OPERATOR_T_ADD) {
                    assigned_right = op::add(left_assign->right_, assigned_right);
                    operator_t = OPERATOR_T_EQL;
                } else if (operator_t == OPERATOR_T_SUB) {
                    assigned_right = op::subtract(left_assign->right_, assigned_right);
                    operator_t = OPERATOR_T_EQL;
                } else if (operator_t == OPERATOR_T_DIV) {
                    assigned_right = op::eltdiv(left_assign->right_, assigned_right);
                    operator_t = OPERATOR_T_EQL;
                } else if (operator_t == OPERATOR_T_MUL) {
                    assigned_right = op::eltmul(left_assign->right_, assigned_right);
                    operator_t = OPERATOR_T_EQL;
                }
            } else {
                // TODO(jonathan): fill this in
                ASSERT2(false, "not sure what to do");
            }
            assigned_left = left_assign->left_;
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
