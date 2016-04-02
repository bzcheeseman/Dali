#include "array.h"

#include "dali/array/typed_array.h"
#include "dali/utils.h"

Array::Array() : TypedArrayVariant(TypedArray<memory_ops::DEVICE_CPU, float>()) {
}

TypedArrayVariant&& dtype_variant(int dtype) {
	if (dtype == 0) {
		return TypedArrayVariant(TypedArray<memory_ops::DEVICE_CPU, float>());
	} else if (dtype == 1) {
		return TypedArrayVariant(TypedArray<memory_ops::DEVICE_CPU, double>());
	} else {
		utils::assert2(false, "beauty is in the eye of the beholder");
		return TypedArrayVariant();
	}
}


Array::Array(int dtype) : TypedArrayVariant(dtype_variant(dtype)) {

}
