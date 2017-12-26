#include "buffer_view.h"

#include <algorithm>

#include "dali/utils/assert2.h"
#include "dali/utils/print_utils.h"
#include "dali/utils/make_message.h"
#include "dali/array/shape.h"
#include "dali/array/array.h"

namespace {
    // shape makes sense
    bool shape_strictly_positive(const std::vector<int>& shape) {
        return std::all_of(shape.begin(), shape.end(), [](int x) {
            return x > 0;
        });
    }
}

std::shared_ptr<memory::SynchronizedMemory> BufferView::create_memory(
        const std::vector<int>& shape,
        DType dtype,
        memory::Device preferred_device) {
    int number_of_elements = hypercube_volume(shape);

    return std::make_shared<memory::SynchronizedMemory>(
        number_of_elements * size_of_dtype(dtype),
        (shape.size() > 0) ? shape[shape.size()-1] : 1,
        preferred_device
    );
}

BufferView::BufferView(std::shared_ptr<memory::SynchronizedMemory> memory,
                       const std::vector<int>& shape,
                       DType dtype,
                       int offset,
                       const std::vector<int>& strides) :
        Expression(shape, dtype, offset, strides),
        memory_(memory){
    ASSERT2(shape_strictly_positive(shape), utils::make_message("Shape "
        "elements must be strictly positive (got ", shape, ")."));
}

BufferView::BufferView(const std::vector<int>& shape,
                       DType dtype,
                       memory::Device preferred_device,
                       int offset,
                       const std::vector<int>& strides) :
        BufferView(create_memory(shape, dtype, preferred_device),
                   shape, dtype, offset, strides) {

}

BufferView::BufferView(const BufferView& other) :
        Expression(other),
        memory_(other.memory_) {
}

expression_ptr BufferView::copy() const {
    return std::make_shared<BufferView>(*this);
}


memory::Device BufferView::preferred_device() const {
    return memory_->preferred_device;
}


std::vector<Array> BufferView::arguments() const {
    return {};
}

bool BufferView::spans_entire_memory() const {
    int noe = number_of_elements();
    if (offset_ == 0 && noe * size_of_dtype(dtype_) == memory_->total_memory) {
        return true;
    }
    if (offset_ == noe - 1) {
        const auto& arr_strides = strides_;
        const auto& arr_shape = shape_;
        for (int i = 0; i < arr_strides.size(); i++) {
            if (std::abs(arr_strides[i]) == 1 && arr_shape[i] == noe) {
                return true;
            }
        }
    }
    return false;
}

std::shared_ptr<BufferView> BufferView::create_with_shape(
        const std::vector<int>& shape,
        DType dtype,
        memory::Device preferred_device,
        const std::vector<int>& broadcasted_axes) {
    auto ret = std::make_shared<BufferView>(shape, dtype, preferred_device);
    for (const auto& axis : broadcasted_axes) {
        ret->broadcast_axis_internal(axis);
    }
    return ret;
}

bool BufferView::supports_operator(OPERATOR_T operator_t) const {
    return true;
}

bool BufferView::is_axis_collapsible_with_axis_minus_one(int axis) const {
    return contiguous_memory();
}

namespace op {
BufferView* static_as_buffer_view(const Array& arr) {
    return static_cast<BufferView*>(arr.expression().get());
}
}
