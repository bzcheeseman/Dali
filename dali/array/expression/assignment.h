#ifndef DALI_ARRAY_EXPRESSION_ASSIGNMENT_H
#define DALI_ARRAY_EXPRESSION_ASSIGNMENT_H

#include "dali/array/array.h"
#include "dali/array/expression/expression.h"

struct Assignment : public Expression {
    Array left_;
    OPERATOR_T operator_t_;
    Array right_;

    virtual expression_ptr copy() const;
    Assignment(Array left, OPERATOR_T operator_t, Array right);
    Assignment(const Assignment& other);
    virtual std::string name() const;
    virtual memory::Device preferred_device() const;
    virtual std::vector<Array> arguments() const;
    virtual bool is_axis_collapsible_with_axis_minus_one(int axis) const;
    virtual expression_ptr collapse_axis_with_axis_minus_one(int axis) const;
};

namespace op {

Array autoreduce_assign(const Array& left, const Array& right);
Array to_assignment(const Array& node);
Array assign(const Array& left, OPERATOR_T, const Array& right);
Assignment* static_as_assignment(const Array& arr);

} // namespace op

#endif  // DALI_ARRAY_EXPRESSION_ASSIGNMENT_H
