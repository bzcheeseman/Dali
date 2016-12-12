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

memory::Device BufferView::preferred_device() const {
    return memory_->preferred_device;
}
