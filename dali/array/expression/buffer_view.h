#ifndef DALI_ARRAY_EXPRESSION_BUFFER_VIEW_H
#define DALI_ARRAY_EXPRESSION_BUFFER_VIEW_H

#include <memory>
#include <vector>

#include "dali/array/expression/expression.h"
#include "dali/array/memory/synchronized_memory.h"

struct BufferView : public Expression {
    std::shared_ptr<memory::SynchronizedMemory> memory_;

    static std::shared_ptr<memory::SynchronizedMemory> create_memory(
            const std::vector<int>& shape,
            DType dtype,
            memory::Device preferred_device);

    BufferView(std::shared_ptr<memory::SynchronizedMemory> memory,
               const std::vector<int>& shape,
               DType dtype,
               int offset,
               const std::vector<int>& strides);

    BufferView(const std::vector<int>& shape,
               DType dtype,
               memory::Device preferred_device,
               int offset=0,
               const std::vector<int>& strides={});

    BufferView(const BufferView& other);

    virtual bool spans_entire_memory() const;
    virtual bool is_assignable() const;
    virtual std::string name() const;

    virtual memory::Device preferred_device() const ;

    static std::shared_ptr<BufferView> create_with_shape(
            const std::vector<int>& shape,
            DType dtype,
            memory::Device preferred_device,
            const std::vector<int>& broadcasted_axes);

    virtual bool supports_operator(OPERATOR_T operator_t) const;
    virtual bool is_axis_collapsible_with_axis_minus_one(int axis) const;
    virtual bool can_copyless_reshape(const std::vector<int>& new_shape) const;
    virtual void broadcast_axis_internal(const int& axis);

    virtual expression_ptr copy() const;
    virtual expression_ptr copy(const std::vector<int>& shape,
                                int offset,
                                const std::vector<int>& strides) const;
    virtual expression_ptr buffer_arg() const;

    virtual expression_ptr dimshuffle(const std::vector<int>& pattern, const Array* owner) const;
    virtual expression_ptr copyless_reshape(const std::vector<int>& new_shape, const Array* owner) const;
    virtual expression_ptr pluck_axis(int axis, const Slice& slice_unnormalized, const Array* owner) const;
    virtual expression_ptr squeeze(int axis, const Array* owner) const;
    virtual expression_ptr expand_dims(int new_axis, const Array* owner) const;
    virtual expression_ptr broadcast_axis(int axis, const Array* owner) const;
    virtual expression_ptr broadcast_to_shape(const std::vector<int>& new_shape, const Array* owner) const;
    virtual expression_ptr reshape(const std::vector<int>& shape, const Array* owner) const;
    virtual expression_ptr operator()(int idx, const Array* owner) const;

};

namespace op {
    BufferView* static_as_buffer_view(const Array& arr);
}  // namespace op


#endif  // DALI_ARRAY_EXPRESSION_BUFFER_VIEW_H
