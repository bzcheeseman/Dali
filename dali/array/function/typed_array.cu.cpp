#include "typed_array.h"

#include "dali/config.h"
#include "dali/utils/print_utils.h"


template<typename T>
T*  TypedArray<memory::DEVICE_T_CPU, T>::ptr(memory::AM access_mode) const {
    return (T*)(array.memory()->data(device, access_mode)) + array.offset();
}

template<typename T>
mshadow::Tensor<mshadow::cpu, 1, T> TypedArray<memory::DEVICE_T_CPU, T>::d1(memory::AM access_mode) const {
    return mshadow::Tensor<mshadow::cpu, 1, T>(ptr(access_mode), mshadow::Shape1(array.number_of_elements()));
}

template<typename T>
TypedArray<memory::DEVICE_T_CPU, T>::TypedArray(const Array& _array, const memory::Device& _device)
        : array(_array), device(_device) {}

template class TypedArray<memory::DEVICE_T_CPU, int>;
template class TypedArray<memory::DEVICE_T_CPU, float>;
template class TypedArray<memory::DEVICE_T_CPU, double>;


#ifdef DALI_USE_CUDA
    template<typename T>
    T* TypedArray<memory::DEVICE_T_GPU, T>::ptr(memory::AM access_mode) const {
        return (T*) ((array.memory())->data(device, access_mode)) + array.offset();;
    }

    template<typename T>
    thrust::device_ptr<T> TypedArray<memory::DEVICE_T_GPU, T>::to_thrust(memory::AM access_mode) const {
        return thrust::device_pointer_cast(ptr(access_mode));
    }

    template<typename T>
    mshadow::Tensor<mshadow::gpu, 1, T> TypedArray<memory::DEVICE_T_GPU, T>::d1(memory::AM access_mode) const {
        return mshadow::Tensor<mshadow::gpu, 1, T>(
            ptr(access_mode),
            mshadow::Shape1(array.number_of_elements())
        );
    }

    template<typename T>
    TypedArray<memory::DEVICE_T_GPU, T>::TypedArray(const Array& _array, const memory::Device& _device)
            : array(_array), device(_device) {}


    template class TypedArray<memory::DEVICE_T_GPU, int>;
    template class TypedArray<memory::DEVICE_T_GPU, float>;
    template class TypedArray<memory::DEVICE_T_GPU, double>;

#endif
