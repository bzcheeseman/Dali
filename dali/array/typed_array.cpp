#include "typed_array.h"

#include "dali/array/memory/device.h"
#include "dali/array/memory/memory_ops.h"
#include "dali/config.h"

template<int dev, typename T>
TypedArray<dev, T>::TypedArray(size_t dim_) : dim(dim_) {
}

template<int dev, typename T>
TypedArray<dev, T>::TypedArray() : TypedArray(0) {
}

template<int dev, typename T>
TypedArray<dev, T> TypedArray<dev, T>::empty_like(const TypedArray<dev, T>& other) {
	return TypedArray<dev, T>(other.dim);
}


#ifdef DALI_USE_CUDA
    template class TypedArray<memory::DEVICE_T_CPU, int>;
    template class TypedArray<memory::DEVICE_T_CPU, float>;
    template class TypedArray<memory::DEVICE_T_CPU, double>;
    template class TypedArray<memory::DEVICE_T_GPU, int>;
    template class TypedArray<memory::DEVICE_T_GPU, float>;
    template class TypedArray<memory::DEVICE_T_GPU, double>;
#else
    template class TypedArray<memory::DEVICE_T_CPU, int>;
    template class TypedArray<memory::DEVICE_T_CPU, float>;
    template class TypedArray<memory::DEVICE_T_CPU, double>;
#endif
