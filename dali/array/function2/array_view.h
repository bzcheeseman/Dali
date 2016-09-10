#ifndef DALI_ARRAY_FUNCTION2_ARRAY_VIEW_H
#define DALI_ARRAY_FUNCTION2_ARRAY_VIEW_H

#include "dali/array/array.h"
#include "dali/array/memory/device.h"
#include "dali/array/shape.h"

template<typename T, int ndim>
class ArrayView {
    public:
        T *const ptr_;
        const int offset_;
        const Shape<ndim> shape_;

        XINLINE ArrayView(T* ptr, int offset, Shape<ndim> shape) :
                ptr_(ptr), offset_(offset), shape_(shape) {
        }

        XINLINE T& operator()(int idx) {
            // assumes contiguous memory
            return *(ptr_ + offset_ + idx);
        }
};

template<typename T, int ndim>
class ArrayStridedView {
    public:
        T *const ptr_;
        const int offset_;
        const Shape<ndim> shape_;
        const Shape<ndim> strides_;

        XINLINE ArrayStridedView(T* ptr, int offset, Shape<ndim> shape, Shape<ndim> strides) :
                ptr_(ptr), offset_(offset), shape_(shape), strides_(strides) {
        }

        XINLINE T& operator()(int idx) {
            // assumes contiguous memory
            return *(ptr_ + offset_ + idx * strides_[0]);
        }
};

template<typename T, int ndim>
ArrayView<T, ndim> make_view(const Array& arr) {
    return ArrayView<T, ndim>((T*)arr.memory()->mutable_data(memory::Device::cpu()),
                              arr.offset(),
                              arr.shape());
}

template<typename T, int ndim>
ArrayStridedView<T, ndim> make_strided_view(const Array& arr) {
    return ArrayStridedView<T, ndim>(
        (T*)arr.memory()->mutable_data(memory::Device::cpu()),
        arr.offset(),
        arr.shape(),
        arr.strides()
    );
}

#endif  // DALI_ARRAY_FUNCTION2_ARRAY_VIEW_H
