#ifndef DALI_ARRAY_FUNCTION2_ARRAY_VIEW_H
#define DALI_ARRAY_FUNCTION2_ARRAY_VIEW_H
#include "dali/macros.h"
#include <initializer_list>

template<int num_dims>
struct Shape {
    int sizes_[num_dims];

    XINLINE Shape() {}

    XINLINE Shape(int value) {
        #pragma unroll
        for (int i = 0; i < num_dims; ++i) {
            sizes_[i] = value;
        }
    }

    XINLINE Shape(const int* sizes) {
        for (int i = 0; i < num_dims;++i) {
            sizes_[i] = sizes[i];
        }
    }

    XINLINE Shape<num_dims + 1> expand_dims(int pos) const {
        Shape<num_dims + 1> new_shape;
        int internal_i = 0;
        #pragma unroll
        for (int i = 0; i < num_dims + 1; ++i) {
            if (i != pos) {
                new_shape[i] = sizes_[internal_i];
                ++internal_i;
            } else {
                new_shape[i] = 1;
            }
        }
        return new_shape;
    }

    template<int start_dim, int newdims, int extra_dimensions=0>
    XINLINE Shape<newdims + extra_dimensions> axis_reduced_shape() const {
        Shape<newdims + extra_dimensions> new_shape;
        #pragma unroll
        for (int i = 0; i < newdims; ++i) {
            new_shape[i + extra_dimensions] = sizes_[start_dim + i];
        }
        return new_shape;
    }

    XINLINE Shape(std::initializer_list<int> sizes) {
        int i = 0;
        for (auto iter = sizes.begin(); iter != sizes.end(); iter++) {
            sizes_[i] = *iter;
            i++;
        }
    }

    XINLINE Shape(const Shape<num_dims>& other) {
        for (int i = 0; i < num_dims; ++i) {
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

template<int dimensions, typename Type>
struct AbstractKernel {
    static const int ndim = dimensions;
    typedef Type T;
    Shape<ndim> shape_;
    const Shape<ndim>& shape() const {
        return shape_;
    }
    AbstractKernel(const Shape<ndim>& shape) : shape_(shape) {}
    virtual T operator[](const Shape<ndim>&) const = 0;
};

int div_ceil(int a, int b) {
    return (a + b - 1) / b;
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

template<typename Type, int dimensions>
class ScalarView {
    public:
        Type scalar_;
        static const int ndim = dimensions;
        const Shape<dimensions> shape_;
        typedef Type T;

        XINLINE ScalarView(const Type& scalar) : scalar_(scalar), shape_(1) {
        }

        XINLINE Type& operator()(int idx) {
            return scalar_;
        }

        XINLINE const Type& operator()(int idx) const {
            return scalar_;
        }

        XINLINE const Shape<dimensions>& shape() const {
            return shape_;
        }

        XINLINE Type& operator[](const Shape<dimensions>& indices) {
            return scalar_;
        }

        XINLINE const Type& operator[](const Shape<dimensions>& indices) const {
            return scalar_;
        }
};

template<typename T, int ndim>
ScalarView<T, ndim> make_scalar_view(const T& scalar) {
    return ScalarView<T, ndim>(scalar);
}

template<typename T, int ndim>
ArrayView<T, ndim> make_view(void* data_ptr, int offset, const int* sizes) {
    return ArrayView<T, ndim>(
        (T*)data_ptr, offset, sizes
    );
}

template<typename T, int ndim>
ArrayStridedView<T, ndim> make_strided_view(void* data_ptr, int offset, const int* sizes, const int* strides) {
    return ArrayStridedView<T, ndim>(
        (T*)data_ptr, offset, sizes, strides
    );
}

#endif  // DALI_ARRAY_FUNCTION2_ARRAY_VIEW_H
