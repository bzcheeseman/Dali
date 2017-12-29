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

    virtual expression_ptr copy() const;
    virtual expression_ptr buffer_arg() const;

    virtual std::vector<Array> arguments() const;

    virtual bool spans_entire_memory() const;
    virtual bool is_assignable() const;

    virtual memory::Device preferred_device() const ;

    static std::shared_ptr<BufferView> create_with_shape(
            const std::vector<int>& shape,
            DType dtype,
            memory::Device preferred_device,
            const std::vector<int>& broadcasted_axes);

    virtual bool supports_operator(OPERATOR_T operator_t) const;
    virtual bool is_axis_collapsible_with_axis_minus_one(int axis) const;
};

namespace op {
    BufferView* static_as_buffer_view(const Array& arr);
}  // namespace op


#endif  // DALI_ARRAY_EXPRESSION_BUFFER_VIEW_H
