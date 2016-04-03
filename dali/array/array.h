#ifndef DALI_ARRAY_ARRAY_H
#define DALI_ARRAY_ARRAY_H

#include <variant.hpp>

#include "dali/array/dtype.h"
#include "dali/array/memory/memory_ops.h"
#include "dali/array/typed_array.h"

// sorry about Syntax.
#ifdef DALI_USE_CUDA
    typedef mapbox::util::variant<TypedArray<memory_ops::DEVICE_CPU, int>,
                                  TypedArray<memory_ops::DEVICE_CPU, float>,
                                  TypedArray<memory_ops::DEVICE_CPU, double>,
                                  TypedArray<memory_ops::DEVICE_GPU, int>,
                                  TypedArray<memory_ops::DEVICE_GPU, float>,
                                  TypedArray<memory_ops::DEVICE_GPU, double>> TypedArrayVariant;
#else
    typedef mapbox::util::variant<TypedArray<memory_ops::DEVICE_CPU, int>,
                                  TypedArray<memory_ops::DEVICE_CPU, float>,
                                  TypedArray<memory_ops::DEVICE_CPU, double>> TypedArrayVariant;
#endif

class Array : public TypedArrayVariant {
    public:
      Array();
      Array(dtype::Dtype dtype_);
      Array(TypedArrayVariant&& typed_array);

};

#endif
