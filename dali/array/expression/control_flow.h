#ifndef DALI_ARRAY_EXPRESSION_CONTROL_FLOW_H
#define DALI_ARRAY_EXPRESSION_CONTROL_FLOW_H

#include <vector>
#include "dali/array/array.h"
#include "dali/array/expression/expression.h"

struct ControlFlow : public Expression {
    const Array& left_;
    ControlFlow(Array left, const std::vector<Array>& conditions);
    ControlFlow(const ControlFlow& other);
    virtual memory::Device preferred_device() const;
    virtual expression_ptr copy() const;
    virtual bool is_axis_collapsible_with_axis_minus_one(int axis) const;
    virtual bool spans_entire_memory() const;
    virtual bool supports_operator(OPERATOR_T operator_t) const;
    virtual bool is_assignable() const;
    virtual expression_ptr buffer_arg() const;
    virtual bool all_conditions_are_met() const;
    virtual std::vector<Array> conditions() const;
};
namespace op {
    ControlFlow* static_as_control_flow(const Array& arr);
    Array control_dependency(Array condition, Array result);
}  // namespace op

#endif  // DALI_ARRAY_EXPRESSION_CONTROL_FLOW_H
