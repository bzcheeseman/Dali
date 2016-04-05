#include "dali/array/getmshadow.h"

template<int devT, typename T>
void MArray<devT, T>::ptr(memory::AM) {}

template<int devT, typename T>
void MArray<devT, T>::d1(memory::AM) {}

template<typename T>
T*  MArray<memory::DEVICE_T_CPU, T>::ptr(memory::AM access_mode) {
    return (T*)(array.memory()->data(device, access_mode)) + array.offset();
}

template<typename T>
mshadow::Tensor<mshadow::cpu, 1, T> MArray<memory::DEVICE_T_CPU, T>::d1(memory::AM access_mode) {
    return mshadow::Tensor<mshadow::cpu, 1, T>(ptr(access_mode), mshadow::Shape1(array.number_of_elements()));
}

template class MArray<memory::DEVICE_T_CPU, int>;
template class MArray<memory::DEVICE_T_CPU, float>;
template class MArray<memory::DEVICE_T_CPU, double>;


#ifdef DALI_USE_CUDA
    template<typename T>
    T* MArray<memory::DEVICE_T_GPU, T>::ptr(memory::AM access_mode) {
        return (T*)(array.memory()->data(device, access_mode)) + array.offset();
    }

    template<typename T>
    thrust::device_ptr<T> MArray<memory::DEVICE_T_GPU, T>::to_thrust(memory::AM access_mode) {
        return thrust::device_pointer_cast(ptr(access_mode));
    }

    template<typename T>
    mshadow::Tensor<mshadow::gpu, 1, T> MArray<memory::DEVICE_T_GPU, T>::d1(memory::AM access_mode) {
        return mshadow::Tensor<mshadow::gpu, 1, T>(
            ptr(access_mode),
            mshadow::Shape1(array.number_of_elements())
        );
    }

    template<> class MArray<memory::DEVICE_T_GPU, int>;
    template<> class MArray<memory::DEVICE_T_GPU, float>;
    template<> class MArray<memory::DEVICE_T_GPU, double>;

#endif
