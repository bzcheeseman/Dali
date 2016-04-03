#ifndef DALI_ARRAY_ARRAY_H
#define DALI_ARRAY_ARRAY_H

#include <variant.hpp>

#include "dali/array/memory/device.h"
#include "dali/array/dtype.h"
#include "dali/array/memory/memory_ops.h"
#include "dali/array/typed_array.h"

// sorry about Syntax.
#ifdef DALI_USE_CUDA
    typedef mapbox::util::variant<TypedArray<memory::DEVICE_T_CPU, int>,
                                  TypedArray<memory::DEVICE_T_CPU, float>,
                                  TypedArray<memory::DEVICE_T_CPU, double>,
                                  TypedArray<memory::DEVICE_T_GPU, int>,
                                  TypedArray<memory::DEVICE_T_GPU, float>,
                                  TypedArray<memory::DEVICE_T_GPU, double>> TypedArrayVariant;
#else
    typedef mapbox::util::variant<TypedArray<memory::DEVICE_T_CPU, int>,
                                  TypedArray<memory::DEVICE_T_CPU, float>,
                                  TypedArray<memory::DEVICE_T_CPU, double>> TypedArrayVariant;
#endif

class Array : public TypedArrayVariant {
    public:
      Array();
      Array(dtype::Dtype dtype_);
      Array(TypedArrayVariant&& typed_array);

};

#endif
