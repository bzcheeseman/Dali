#include "assignment.h"
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
    return "Assignment[" + operator_to_name(operator_t_) + "]";
}


Array autoreduce_assign(const Array& left, const Array& right) {
    throw std::runtime_error("autoreduce_assign not implemented yet.");
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
        // a temp is added so that non overwriting operators
        // can be run independently from the right side's evaluation.
        return Array(std::make_shared<Assignment>(left, operator_t, to_assignment(right)));
    }
}

std::shared_ptr<Assignment> as_assignment(const Array& arr) {
    return std::dynamic_pointer_cast<Assignment>(arr.expression());
}
