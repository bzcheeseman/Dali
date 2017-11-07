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
