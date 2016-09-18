#ifndef DALI_ARRAY_FUNCTION2_ARRAY_VIEW_H
#define DALI_ARRAY_FUNCTION2_ARRAY_VIEW_H

#include "dali/array/array.h"
#include "dali/array/memory/device.h"
#include "dali/macros.h"
#include <vector>

template<int num_dims>
struct Shape {
    int sizes_[num_dims];

    XINLINE Shape() {}

    Shape(const std::vector<int>& sizes) {
        for (int i = 0; i < sizes.size();i++) {
            sizes_[i] = sizes[i];
        }
    }

    XINLINE Shape(std::initializer_list<int> sizes) {
        int i = 0;
        for (auto iter = sizes.begin(); iter != sizes.end(); iter++) {
            sizes_[i] = *iter;
            i++;
        }
    }

    XINLINE Shape(const Shape<num_dims>& other) {
        for (int i = 0; i < num_dims; i++) {
            sizes_[i] = other.sizes_[i];
        }
    }

    XINLINE ~Shape() {}

    XINLINE int ndim() const {
        return num_dims;
    }

    XINLINE const int& operator[](int dim) const {
        return sizes_[dim];
    }

    XINLINE int& operator[](int dim) {
        return sizes_[dim];
    }

    void XINLINE set_dim(int dim, int value) {
        sizes_[dim] = value;
    }

    XINLINE Shape& operator=(const Shape<num_dims>& other) {
        for (int i = 0; i < other.ndim(); i++) {
            sizes_[i] = other[i];
        }
        return *this;
    }

    int XINLINE numel() const {
        int volume = 1;
        for (int i = 0; i < num_dims; i++) {
            volume *= sizes_[i];
        }
        return volume;
    }
};

template<int ndim>
XINLINE Shape<ndim> index_to_dim(int index, const Shape<ndim>& shape) {
    Shape<ndim> multi_dimensional_index;
    #pragma unroll
    for (int i = 0; i < ndim; i++) {
        multi_dimensional_index[i] = index % shape[i];
        index /= shape[i];
    }
    return multi_dimensional_index;
}

template<int ndim>
XINLINE int indices_to_offset(const Shape<ndim>& shape, const Shape<ndim>& indices) {
    int offset = 0;
    int volume = 1;
    #pragma unroll
    for (int i = ndim - 1; i >= 0; --i) {
        offset += volume * indices[i];
        volume *= shape[i];
    }
    return offset;
}

template<int ndim>
XINLINE int indices_to_offset(const Shape<ndim>& shape, const Shape<ndim>& indices, const Shape<ndim>& strides) {
    int offset = 0;
    #pragma unroll
    for (int i = 0; i < ndim; i++) {
        offset += strides[i] * indices[i];
    }
    return offset;
}

// assumes contiguous memory
template<typename Type, int dimensions>
class ArrayView {
    public:
        Type *const ptr_;
        const int offset_;
        const Shape<dimensions> shape_;

        static const int ndim = dimensions;
        typedef Type T;

        XINLINE ArrayView(Type* ptr, int offset, Shape<dimensions> shape) :
                ptr_(ptr), offset_(offset), shape_(shape) {
        }

        XINLINE Type& operator()(int idx) {
            return *(ptr_ + offset_ + idx);
        }

        XINLINE const Type& operator()(int idx) const {
            return *(ptr_ + offset_ + idx);
        }

        XINLINE const Shape<dimensions>& shape() const {
            return shape_;
        }

        XINLINE Type& operator[](const Shape<dimensions>& indices) {
            return *(ptr_ + offset_ + indices_to_offset(shape_, indices));
        }

        XINLINE const Type& operator[](const Shape<dimensions>& indices) const {
            return *(ptr_ + offset_ + indices_to_offset(shape_, indices));
        }
};

// assumes strided memory
template<typename Type, int dimensions>
class ArrayStridedView {
    public:
        Type *const ptr_;
        const int offset_;
        const Shape<dimensions> shape_;
        const Shape<dimensions> strides_;

        static const int ndim = dimensions;
        typedef Type T;

        XINLINE ArrayStridedView(Type* ptr, int offset, Shape<dimensions> shape, Shape<dimensions> strides) :
                ptr_(ptr), offset_(offset), shape_(shape), strides_(strides) {
        }

        XINLINE Type& operator()(int idx) {
            return *(ptr_ + offset_ + idx * strides_[0]);
        }

        XINLINE const Type& operator()(int idx) const {
            return *(ptr_ + offset_ + idx * strides_[0]);
        }

        XINLINE const Shape<dimensions>& shape() const {
            return shape_;
        }

        XINLINE Type& operator[](const Shape<dimensions>& indices) {
            return *(ptr_ + offset_ + indices_to_offset(shape_, strides_, indices));
        }

        XINLINE const Type& operator[](const Shape<dimensions>& indices) const {
            return *(ptr_ + offset_ + indices_to_offset(shape_, strides_, indices));
        }
};

template<typename T, int ndim>
ArrayView<T, ndim> make_view(const Array& arr) {
    return ArrayView<T, ndim>(
        (T*)arr.memory()->mutable_data(memory::Device::cpu()),
        arr.offset(),
        arr.shape()
    );
}

template<typename T, int ndim>
ArrayStridedView<T, ndim> make_strided_view(const Array& arr) {
    return ArrayStridedView<T, ndim>(
        (T*)arr.memory()->mutable_data(memory::Device::cpu()),
        arr.offset(),
        arr.shape(),
        arr.strides()
    );
}

#endif  // DALI_ARRAY_FUNCTION2_ARRAY_VIEW_H
