#include "array.h"

#include "dali/array/typed_array.h"
#include "dali/utils.h"

template< dtype::Dtype dtype >
struct always_false {
    enum { value = false };
};

TypedArrayVariant&& dtype_variant(dtype::Dtype&& dtype_) {
    if (dtype_ == dtype::Float) {
        return TypedArrayVariant(TypedArray<memory::DEVICE_T_CPU, float>());
    } else if (dtype_ == dtype::Double) {
        return TypedArrayVariant(TypedArray<memory::DEVICE_T_CPU, double>());
    } else if (dtype_ == dtype::Int32) {
        return TypedArrayVariant(TypedArray<memory::DEVICE_T_CPU, int>());
    } else {
        utils::assert2(false, "TypedArray can only be of type " DALI_ACCEPTABLE_DTYPE_STR ".");
    }
    return TypedArrayVariant();
}

Array::Array() : Array(dtype::Float) {}

Array::Array(dtype::Dtype dtype_) : TypedArrayVariant(dtype_variant(std::move(dtype_))) {}

Array::Array(TypedArrayVariant&& typed_array) : TypedArrayVariant(std::move(typed_array)) {}
