#include "typed_array.h"

#include "dali/config.h"
#include "dali/utils/print_utils.h"


namespace internal {
    template<typename MDevT, typename T>
    T* TypedArrayShared<MDevT,T>::ptr_internal(memory::AM access_mode) const {
        return (T*)(array.memory()->data(device, access_mode)) + array.offset();
    }

    template<typename MDevT, typename T>
    T* TypedArrayShared<MDevT,T>::ptr(memory::AM access_mode) const {
        ASSERT2(this->array.contiguous_memory(),
                "This function is only supported for contiguous_memory");
        return ptr_internal(access_mode);
    }

    template<typename MDevT, typename T>
    TypedArrayShared<MDevT, T>::TypedArrayShared(const Array& _array, const memory::Device& _device, const std::vector<int>& _output_shape)
            : array(_array.reshape_broadcasted(_output_shape)), device(_device) {
    }

    template<typename MDevT, typename T>
    std::tuple<bool,mshadow::Tensor<MDevT, 2, T>> TypedArrayShared<MDevT, T>::blas_friendly_tensor() const {
        ASSERT2(array.ndim() == 2,
                utils::MS() << "blas_friendly_tensor is only available to 2D tensors ("
                            << array.ndim() << "D tensor passed.)");
        if (array.strides().size() == 0) {
            return std::make_tuple(false, mtensor<2>());
        }

        const std::vector<int>& a_strides = array.strides();

        if (a_strides[0] == 1) {
            auto ret = mtensor<2>();
            ret.stride_ = a_strides[1];
            return std::make_tuple(true, ret);
        } else if (a_strides[1] == 1) {
            auto ret = mtensor<2>();
            ret.stride_ = a_strides[0];
            return std::make_tuple(false, ret);
        } else {
            ASSERT2(a_strides[0] == 1 || a_strides[1] == 1,
                    utils::MS() << "gemm does not support doubly strided matrices (input strides: " << a_strides << ")");
            return std::make_tuple(false, mtensor<2>());
        }
    }



    template<typename MDevT, typename T>
    mshadow::Tensor<MDevT, 1, T> TypedArrayShared<MDevT,T>::contiguous_d1(memory::AM access_mode, bool collapse_leading) const { return contiguous_d<1>(access_mode, collapse_leading); }
    template<typename MDevT, typename T>
    mshadow::Tensor<MDevT, 2, T> TypedArrayShared<MDevT,T>::contiguous_d2(memory::AM access_mode, bool collapse_leading) const { return contiguous_d<2>(access_mode, collapse_leading); }
    template<typename MDevT, typename T>
    mshadow::Tensor<MDevT, 3, T> TypedArrayShared<MDevT,T>::contiguous_d3(memory::AM access_mode, bool collapse_leading) const { return contiguous_d<3>(access_mode, collapse_leading); }
    template<typename MDevT, typename T>
    mshadow::Tensor<MDevT, 4, T> TypedArrayShared<MDevT,T>::contiguous_d4(memory::AM access_mode, bool collapse_leading) const { return contiguous_d<4>(access_mode, collapse_leading); }

    template<typename MDevT, typename T>
    DaliWrapperExp<MDevT, 1, T> TypedArrayShared<MDevT,T>::d1(memory::AM access_mode, bool collapse_leading) const { return d<1>(access_mode, collapse_leading); }
    template<typename MDevT, typename T>
    DaliWrapperExp<MDevT, 2, T> TypedArrayShared<MDevT,T>::d2(memory::AM access_mode, bool collapse_leading) const { return d<2>(access_mode, collapse_leading); }
    template<typename MDevT, typename T>
    DaliWrapperExp<MDevT, 3, T> TypedArrayShared<MDevT,T>::d3(memory::AM access_mode, bool collapse_leading) const { return d<3>(access_mode, collapse_leading); }
    template<typename MDevT, typename T>
    DaliWrapperExp<MDevT, 4, T> TypedArrayShared<MDevT,T>::d4(memory::AM access_mode, bool collapse_leading) const { return d<4>(access_mode, collapse_leading); }


    template class TypedArrayShared<mshadow::cpu, int>;
    template class TypedArrayShared<mshadow::cpu, float>;
    template class TypedArrayShared<mshadow::cpu, double>;
    #ifdef DALI_USE_CUDA
        template class TypedArrayShared<mshadow::gpu, int>;
        template class TypedArrayShared<mshadow::gpu, float>;
        template class TypedArrayShared<mshadow::gpu, double>;
    #endif

    // Subtensor

    template<typename MDevT, typename T, typename IndexT>
    TypedArraySubtensorShared<MDevT, T, IndexT>::TypedArraySubtensorShared(const Array& _source, const Array& _indices, const memory::Device& _device, const std::vector<int>& _output_shape)
            : source(_source, _device, _output_shape), indices(_indices, _device, _output_shape) {
    }

    template<typename MDevT, typename T, typename IndexT>
    mshadow::expr::TakeFromRowsExp<mshadow::Tensor<MDevT, 0, IndexT>, mshadow::Tensor<MDevT, 1, T>, T, IndexT> TypedArraySubtensorShared<MDevT,T, IndexT>::contiguous_d1(memory::AM access_mode, bool collapse_leading) const { return contiguous_d<1>(access_mode, collapse_leading); }
    template<typename MDevT, typename T, typename IndexT>
    mshadow::expr::TakeFromRowsExp<mshadow::Tensor<MDevT, 1, IndexT>, mshadow::Tensor<MDevT, 2, T>, T, IndexT> TypedArraySubtensorShared<MDevT,T, IndexT>::contiguous_d2(memory::AM access_mode, bool collapse_leading) const { return contiguous_d<2>(access_mode, collapse_leading); }
    template<typename MDevT, typename T, typename IndexT>
    mshadow::expr::TakeFromRowsExp<mshadow::Tensor<MDevT, 2, IndexT>, mshadow::Tensor<MDevT, 3, T>, T, IndexT> TypedArraySubtensorShared<MDevT,T, IndexT>::contiguous_d3(memory::AM access_mode, bool collapse_leading) const { return contiguous_d<3>(access_mode, collapse_leading); }
    template<typename MDevT, typename T, typename IndexT>
    mshadow::expr::TakeFromRowsExp<mshadow::Tensor<MDevT, 3, IndexT>, mshadow::Tensor<MDevT, 4, T>, T, IndexT> TypedArraySubtensorShared<MDevT,T, IndexT>::contiguous_d4(memory::AM access_mode, bool collapse_leading) const { return contiguous_d<4>(access_mode, collapse_leading); }

    template<typename MDevT, typename T, typename IndexT>
    mshadow::expr::TakeFromRowsExp<DaliWrapperExp<MDevT, 0, IndexT>, DaliWrapperExp<MDevT, 1, T>, T, IndexT> TypedArraySubtensorShared<MDevT,T, IndexT>::d1(memory::AM access_mode, bool collapse_leading) const { return d<1>(access_mode, collapse_leading); }
    template<typename MDevT, typename T, typename IndexT>
    mshadow::expr::TakeFromRowsExp<DaliWrapperExp<MDevT, 1, IndexT>, DaliWrapperExp<MDevT, 2, T>, T, IndexT> TypedArraySubtensorShared<MDevT,T, IndexT>::d2(memory::AM access_mode, bool collapse_leading) const { return d<2>(access_mode, collapse_leading); }
    template<typename MDevT, typename T, typename IndexT>
    mshadow::expr::TakeFromRowsExp<DaliWrapperExp<MDevT, 2, IndexT>, DaliWrapperExp<MDevT, 3, T>, T, IndexT> TypedArraySubtensorShared<MDevT,T, IndexT>::d3(memory::AM access_mode, bool collapse_leading) const { return d<3>(access_mode, collapse_leading); }
    template<typename MDevT, typename T, typename IndexT>
    mshadow::expr::TakeFromRowsExp<DaliWrapperExp<MDevT, 3, IndexT>, DaliWrapperExp<MDevT, 4, T>, T, IndexT> TypedArraySubtensorShared<MDevT,T, IndexT>::d4(memory::AM access_mode, bool collapse_leading) const { return d<4>(access_mode, collapse_leading); }

    template class TypedArraySubtensorShared<mshadow::cpu, int, int>;
    template class TypedArraySubtensorShared<mshadow::cpu, float, int>;
    template class TypedArraySubtensorShared<mshadow::cpu, double, int>;
    #ifdef DALI_USE_CUDA
        template class TypedArraySubtensorShared<mshadow::gpu, int, int>;
        template class TypedArraySubtensorShared<mshadow::gpu, float, int>;
        template class TypedArraySubtensorShared<mshadow::gpu, double, int>;
    #endif
} // namespace internal

template class TypedArray<memory::DEVICE_T_CPU, int>;
template class TypedArray<memory::DEVICE_T_CPU, float>;
template class TypedArray<memory::DEVICE_T_CPU, double>;

template class TypedArraySubtensor<memory::DEVICE_T_CPU, int, int>;
template class TypedArraySubtensor<memory::DEVICE_T_CPU, float, int>;
template class TypedArraySubtensor<memory::DEVICE_T_CPU, double, int>;


#ifdef DALI_USE_CUDA
    template<typename T>
    thrust::device_ptr<T> TypedArray<memory::DEVICE_T_GPU, T>::to_thrust(memory::AM access_mode) const {
        return thrust::device_pointer_cast(this->ptr(access_mode));
    }

    template class TypedArray<memory::DEVICE_T_GPU, int>;
    template class TypedArray<memory::DEVICE_T_GPU, float>;
    template class TypedArray<memory::DEVICE_T_GPU, double>;

    template class TypedArraySubtensor<memory::DEVICE_T_GPU, int, int>;
    template class TypedArraySubtensor<memory::DEVICE_T_GPU, float, int>;
    template class TypedArraySubtensor<memory::DEVICE_T_GPU, double, int>;

#endif
