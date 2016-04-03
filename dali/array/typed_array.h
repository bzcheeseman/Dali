#ifndef DALI_ARRAY_TYPED_ARRAY_H
#define DALI_ARRAY_TYPED_ARRAY_H

#include <cstddef>

template<int dev, typename T>
class TypedArray {
    public:
        size_t dim;

        TypedArray();
        TypedArray(size_t dim_);

        static TypedArray empty_like(const TypedArray<dev, T>& other);
};

#endif
