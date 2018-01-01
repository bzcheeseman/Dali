#ifndef DALI_ARRAY_EXPRESSION_EXPRESSION_H
#define DALI_ARRAY_EXPRESSION_EXPRESSION_H

#include <memory>
#include <vector>

#include "dali/array/dtype.h"
#include "dali/array/slice.h"
#include "dali/array/memory/device.h"
#include "dali/array/expression/operator.h"

struct Expression;
typedef std::shared_ptr<Expression> expression_ptr;

class Array;

struct Expression {
  public:
    std::vector<int> shape_;
    DType            dtype_;
    std::vector<int> strides_;
    int              offset_; // expressing in number of numbers (not bytes)
    const std::vector<Array> arguments_;

    // implemented these for all expression subclasses
    virtual memory::Device preferred_device() const = 0;
    virtual expression_ptr copy() const = 0;

    const std::vector<Array>& arguments() const;
    Expression(const std::vector<int>& shape,
               DType dtype,
               const std::vector<Array>& arguments,
               int offset=0,
               const std::vector<int>& strides={});

    Expression(const Expression& other);

    virtual expression_ptr copy(const std::vector<int>& shape,
                                int offset,
                                const std::vector<int>& strides) const;

    virtual int number_of_elements() const;
    int ndim() const;
    std::vector<int> normalized_strides() const;

    bool is_scalar() const;
    bool is_vector() const;
    bool is_matrix() const;

    virtual bool contiguous_memory() const;
    bool can_copyless_reshape(const std::vector<int>& new_shape) const;
    virtual int normalize_axis(const int& axis) const;
    void broadcast_axis_internal(const int& axis);

    virtual expression_ptr operator()(int idx) const;
    virtual bool is_transpose() const;

    virtual expression_ptr buffer_arg() const;
    virtual expression_ptr broadcast_to_shape(const std::vector<int>& shape) const;
    virtual expression_ptr transpose() const;
    virtual expression_ptr transpose(const std::vector<int>& axes) const;
    virtual expression_ptr swapaxes(int axis1, int axis2) const;
    virtual expression_ptr dimshuffle(const std::vector<int>& pattern) const;
    virtual expression_ptr copyless_ravel() const;
    virtual expression_ptr ravel() const;
    virtual expression_ptr copyless_reshape(const std::vector<int>& shape) const;
    virtual expression_ptr right_fit_ndim(int dimensionality) const;
    virtual expression_ptr copyless_right_fit_ndim(int dimensionality) const;
    virtual expression_ptr reshape(const std::vector<int>& shape) const;
    virtual expression_ptr pluck_axis(int axis, const Slice& slice) const;
    virtual expression_ptr pluck_axis(const int& axis, const int& idx) const;
    virtual expression_ptr squeeze(int axis) const;
    virtual expression_ptr expand_dims(int new_axis) const;
    virtual expression_ptr broadcast_axis(int axis) const;
    virtual expression_ptr insert_broadcast_axis(int new_axis) const;
    virtual expression_ptr broadcast_scalar_to_ndim(const int& ndim) const;
    virtual expression_ptr collapse_axis_with_axis_minus_one(int axis) const;
    virtual bool is_axis_collapsible_with_axis_minus_one(int axis) const;
    virtual std::string name() const;
    virtual std::string full_name() const;
    virtual std::string pretty_print_full_name(Expression* highlight, int indent=0) const;
    virtual bool supports_operator(OPERATOR_T operator_t) const;
    virtual bool spans_entire_memory() const;
    virtual bool is_assignable() const;

    void for_all_suboperations(std::function<void(const Array&)>) const;
};

#endif  // DALI_ARRAY_EXPRESSION_EXPRESSION_H
