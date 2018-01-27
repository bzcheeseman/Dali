#ifndef DALI_ARRAY_EXPRESSION_ASSIGNMENT_H
#define DALI_ARRAY_EXPRESSION_ASSIGNMENT_H

#include "dali/array/array.h"
#include "dali/array/expression/expression.h"

struct Assignment : public Expression {
    OPERATOR_T operator_t_;
    Assignment(Array left, OPERATOR_T operator_t, Array right);
    Assignment(const Assignment& other);

    const Array& left() const;
    const Array& right() const;

    virtual expression_ptr copy() const override;
    virtual std::string name() const override;
    virtual memory::Device preferred_device() const override;
    virtual bool is_axis_collapsible_with_axis_minus_one(int axis) const override;
    virtual expression_ptr collapse_axis_with_axis_minus_one(int axis, const Array* owner) const override;
    virtual bool spans_entire_memory() const override;
    virtual bool is_assignable() const override;
    virtual expression_ptr buffer_arg() const override;

    virtual expression_ptr dimshuffle(const std::vector<int>& pattern, const Array* owner) const override;
    virtual expression_ptr pluck_axis(int axis, const Slice& slice_unnormalized, const Array* owner) const override;
    virtual expression_ptr broadcast_to_shape(const std::vector<int>& new_shape, const Array* owner) const override;
    virtual expression_ptr operator()(int idx, const Array* owner) const override;

    virtual expression_ptr _reshape(const std::vector<int>& shape, const Array* owner) const override;
    virtual expression_ptr _expand_dims(int new_axis, const Array* owner) const override;
    virtual expression_ptr _squeeze(int axis, const Array* owner) const override;

};

namespace op {

Array autoreduce_assign(const Array& left, const Array& right);
Array to_assignment(const Array& node);
Array assign(const Array& left, OPERATOR_T, const Array& right);
Assignment* static_as_assignment(const Array& arr);

} // namespace op

#endif  // DALI_ARRAY_EXPRESSION_ASSIGNMENT_H
