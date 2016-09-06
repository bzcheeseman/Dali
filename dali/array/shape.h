#ifndef DALI_ARRAY_SHAPE_H
#define DALI_ARRAY_SHAPE_H

#include "dali/macros.h"

#include <iostream>
#include <vector>

int hypercube_volume(const std::vector<int>& shape);
std::vector<int> bshape2shape(const std::vector<int>& bshape);
std::vector<int> shape_to_trivial_strides(const std::vector<int>& shape);

template<int num_dims>
struct Shape {
    int sizes_[num_dims];

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

    int XINLINE ndim() const {
        return num_dims;
    }

    int XINLINE operator[](int dim) const {
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

    static int XINLINE index2offset(const Shape<num_dims>& sizes, const Shape<num_dims>& indices) {
        int offset = 0;
        int volume = 1;
        for (int i = indices.ndim() - 1; i >= 0; i--) {
            offset += indices[i] * volume;
            volume *= sizes[i];
        }
        return offset;
    }
};


template<int num_dims>
std::ostream& operator<<(std::ostream& stream, const Shape<num_dims>& dims) {
    stream << "(";
    for (int i = 0; i < dims.ndim();i++) {
        stream << dims[i];
        if (i != dims.ndim() - 1) {
            stream << ", ";
        } else {
            stream << ")";
        }
    }
    return stream;
}

#endif  // DALI_ARRAY_SHAPE_H
