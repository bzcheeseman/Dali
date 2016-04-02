#include "typed_array.h"

#include "dali/array/memory/memory_ops.h"
#include "dali/config.h"

template<int dev, typename T>
TypedArray<dev, T>::TypedArray(int dim) {
}

template<int dev, typename T>
TypedArray<dev, T>::TypedArray() : TypedArray(0) {
}


#ifdef DALI_USE_CUDA
    template class TypedArray<memory_ops::DEVICE_CPU, int>;
    template class TypedArray<memory_ops::DEVICE_CPU, float>;
    template class TypedArray<memory_ops::DEVICE_CPU, double>;
    template class TypedArray<memory_ops::DEVICE_GPU, int>;
    template class TypedArray<memory_ops::DEVICE_GPU, float>;
    template class TypedArray<memory_ops::DEVICE_GPU, double>;
#else
    template class TypedArray<memory_ops::DEVICE_CPU, int>;
    template class TypedArray<memory_ops::DEVICE_CPU, float>;
    template class TypedArray<memory_ops::DEVICE_CPU, double>;
#endif
