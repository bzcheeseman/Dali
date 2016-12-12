#include "buffer_view.h"

#include "dali/utils/assert2.h"
#include "dali/utils/make_message.h"
#include "dali/utils/print_utils.h"
#include "dali/array/shape.h"

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

std::shared_ptr<BufferView> BufferView::construct_with_bshape(
        const std::vector<int>& bshape,
        DType dtype,
        memory::Device preferred_device) {
    auto ret = std::make_shared<BufferView>(
            bshape2shape(bshape), dtype, preferred_device);
    for (int i = 0; i < bshape.size(); ++i) {
        if (bshape[i] < 0) {
            ASSERT2(bshape[i] == -1, "Currently only one-sized broadcasting "
                "is supported.");
            ret->broadcast_axis_internal(i);
        }
    }

    return ret;
}
