#ifndef DALI_ARRAY_EXPRESSION_CONTROL_FLOW_H
#define DALI_ARRAY_EXPRESSION_CONTROL_FLOW_H

#include <vector>
#include "dali/array/array.h"
#include "dali/array/expression/expression.h"

struct ControlFlow : public Expression {
    const Array& left() const;
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

    virtual expression_ptr dimshuffle(const std::vector<int>& pattern, const Array* owner) const override;
    virtual expression_ptr pluck_axis(int axis, const Slice& slice_unnormalized, const Array* owner) const override;
    virtual expression_ptr squeeze(int axis, const Array* owner) const override;
    virtual expression_ptr expand_dims(int new_axis, const Array* owner) const override;
    virtual expression_ptr broadcast_axis(int axis, const Array* owner) const override;
    virtual expression_ptr broadcast_to_shape(const std::vector<int>& new_shape, const Array* owner) const override;
    virtual expression_ptr _reshape(const std::vector<int>& shape, const Array* owner) const override;
    virtual expression_ptr operator()(int idx, const Array* owner) const override;
};
namespace op {
    ControlFlow* static_as_control_flow(const Array& arr);
    Array control_dependency(Array condition, Array result);
}  // namespace op

#endif  // DALI_ARRAY_EXPRESSION_CONTROL_FLOW_H
