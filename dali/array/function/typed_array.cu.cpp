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
        return ptr_internal(access_mode);
    }

    template<typename MDevT, typename T>
    bool TypedArrayShared<MDevT,T>::contiguous_memory() const {
        return this->array.contiguous_memory();
    }

    template<typename MDevT, typename T>
    bool TypedArrayShared<MDevT,T>::spans_entire_memory() const {
        return this->array.spans_entire_memory();
    }

    template<typename MDevT, typename T>
    TypedArrayShared<MDevT, T>::TypedArrayShared(const Array& _array, const memory::Device& _device, const std::vector<int>& _output_shape)
            : array(_array.reshape_broadcasted(_output_shape)), device(_device) {
        ASSERT2(template_to_dtype<T>() == _array.dtype(),
            utils::MS() << "TypedArray got a wrong type (array.dtype()="
                        << _array.dtype() << " vs. TypedArray's given dtype ="
                        << template_to_dtype<T>() << ")."
        );
    }

    template<typename MDevT, typename T>
    std::tuple<bool,mshadow::Tensor<MDevT, 2, T>> TypedArrayShared<MDevT, T>::blas_friendly_tensor(
            memory::AM access_mode, bool collapse_leading) const {
        ASSERT2(array.ndim() == 2,
                utils::MS() << "blas_friendly_tensor is only available to 2D tensors ("
                            << array.ndim() << "D tensor passed.)");
        if (array.strides().size() == 0) {
            return std::make_tuple(false, mtensor<2>(access_mode, collapse_leading));
        }

        const std::vector<int>& a_strides = array.strides();

        if (a_strides[0] == 1) {
            auto ret = mtensor<2>(access_mode, collapse_leading);
            ret.stride_ = a_strides[1];
            return std::make_tuple(true, ret);
        } else if (a_strides[1] == 1) {
            auto ret = mtensor<2>(access_mode, collapse_leading);
            ret.stride_ = a_strides[0];
            return std::make_tuple(false, ret);
        } else {
            ASSERT2(a_strides[0] == 1 || a_strides[1] == 1,
                    utils::MS() << "gemm does not support doubly strided matrices (input strides: " << a_strides << ")");
            return std::make_tuple(false, mtensor<2>(access_mode, collapse_leading));
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
    TypedArraySubtensorShared<MDevT, T, IndexT>::TypedArraySubtensorShared(const Array& _source, const Array& _indices, const std::vector<int> _shape, const memory::Device& _device)
            : source(_source, _device, _source.shape()),
              indices(_indices, _device, _indices.shape()),
              shape(_shape),
              device(_device) {
        ASSERT2(template_to_dtype<T>() == _source.dtype(),
            utils::MS() << "TypedArraySubtensor got a wrong type (array.dtype()="
                        << _source.dtype() << " vs. TypedArraySubtensor's given dtype ="
                        << template_to_dtype<T>() << ")."
        );
        ASSERT2(template_to_dtype<IndexT>() == _indices.dtype(),
            utils::MS() << "TypedArraySubtensor got a wrong type (indices.dtype()="
                        << _indices.dtype() << " vs. TypedArraySubtensor's indices given dtype ="
                        << template_to_dtype<IndexT>() << ")."
        );
    }

    template<typename MDevT, typename T, typename IndexT>
    bool TypedArraySubtensorShared<MDevT,T,IndexT>::contiguous_memory() const {
        return this->source.contiguous_memory() && this->indices.contiguous_memory();
    }

    template<typename MDevT, typename T, typename IndexT>
    bool TypedArraySubtensorShared<MDevT,T,IndexT>::spans_entire_memory() const {
        return this->source.spans_entire_memory() && this->indices.spans_entire_memory();
    }


    template<typename MDevT, typename T, typename IndexT>
    mshadow::expr::GatherFromRows<mshadow::Tensor<MDevT, 1, IndexT>, mshadow::Tensor<MDevT, 2, T>, T, IndexT> TypedArraySubtensorShared<MDevT,T, IndexT>::contiguous_d1(memory::AM access_mode, bool collapse_leading) const { return contiguous_d<1>(access_mode, collapse_leading); }
    template<typename MDevT, typename T, typename IndexT>
    mshadow::expr::GatherFromRows<mshadow::Tensor<MDevT, 2, IndexT>, mshadow::Tensor<MDevT, 3, T>, T, IndexT> TypedArraySubtensorShared<MDevT,T, IndexT>::contiguous_d2(memory::AM access_mode, bool collapse_leading) const { return contiguous_d<2>(access_mode, collapse_leading); }
    template<typename MDevT, typename T, typename IndexT>
    mshadow::expr::GatherFromRows<mshadow::Tensor<MDevT, 3, IndexT>, mshadow::Tensor<MDevT, 4, T>, T, IndexT> TypedArraySubtensorShared<MDevT,T, IndexT>::contiguous_d3(memory::AM access_mode, bool collapse_leading) const { return contiguous_d<3>(access_mode, collapse_leading); }
    template<typename MDevT, typename T, typename IndexT>
    mshadow::expr::GatherFromRows<mshadow::Tensor<MDevT, 4, IndexT>, mshadow::Tensor<MDevT, 5, T>, T, IndexT> TypedArraySubtensorShared<MDevT,T, IndexT>::contiguous_d4(memory::AM access_mode, bool collapse_leading) const { return contiguous_d<4>(access_mode, collapse_leading); }

    template<typename MDevT, typename T, typename IndexT>
    mshadow::expr::GatherFromRows<DaliWrapperExp<MDevT, 1, IndexT>, DaliWrapperExp<MDevT, 2, T>, T, IndexT> TypedArraySubtensorShared<MDevT,T, IndexT>::d1(memory::AM access_mode, bool collapse_leading) const { return d<1>(access_mode, collapse_leading); }
    template<typename MDevT, typename T, typename IndexT>
    mshadow::expr::GatherFromRows<DaliWrapperExp<MDevT, 2, IndexT>, DaliWrapperExp<MDevT, 3, T>, T, IndexT> TypedArraySubtensorShared<MDevT,T, IndexT>::d2(memory::AM access_mode, bool collapse_leading) const { return d<2>(access_mode, collapse_leading); }
    template<typename MDevT, typename T, typename IndexT>
    mshadow::expr::GatherFromRows<DaliWrapperExp<MDevT, 3, IndexT>, DaliWrapperExp<MDevT, 4, T>, T, IndexT> TypedArraySubtensorShared<MDevT,T, IndexT>::d3(memory::AM access_mode, bool collapse_leading) const { return d<3>(access_mode, collapse_leading); }
    template<typename MDevT, typename T, typename IndexT>
    mshadow::expr::GatherFromRows<DaliWrapperExp<MDevT, 4, IndexT>, DaliWrapperExp<MDevT, 5, T>, T, IndexT> TypedArraySubtensorShared<MDevT,T, IndexT>::d4(memory::AM access_mode, bool collapse_leading) const { return d<4>(access_mode, collapse_leading); }

    template class TypedArraySubtensorShared<mshadow::cpu, int, int>;
    template class TypedArraySubtensorShared<mshadow::cpu, float, int>;
    template class TypedArraySubtensorShared<mshadow::cpu, double, int>;
    #ifdef DALI_USE_CUDA
        template class TypedArraySubtensorShared<mshadow::gpu, int, int>;
        template class TypedArraySubtensorShared<mshadow::gpu, float, int>;
        template class TypedArraySubtensorShared<mshadow::gpu, double, int>;
    #endif


    // ArrayGather
    template<typename MDevT, typename T, typename IndexT>
    TypedArrayGatherShared<MDevT, T, IndexT>::TypedArrayGatherShared(const Array& _source, const Array& _indices, const std::vector<int> _shape, const memory::Device& _device)
            : source(_source, _device, _source.shape()),
              indices(_indices, _device, _indices.shape()),
              shape(_shape),
              device(_device) {
        ASSERT2(template_to_dtype<T>() == _source.dtype(),
            utils::MS() << "TypedArraySubtensor got a wrong type (array.dtype()="
                        << _source.dtype() << " vs. TypedArraySubtensor's given dtype ="
                        << template_to_dtype<T>() << ")."
        );
        ASSERT2(template_to_dtype<IndexT>() == _indices.dtype(),
            utils::MS() << "TypedArraySubtensor got a wrong type (indices.dtype()="
                        << _indices.dtype() << " vs. TypedArraySubtensor's indices given dtype ="
                        << template_to_dtype<IndexT>() << ")."
        );
    }

    template<typename MDevT, typename T, typename IndexT>
    bool TypedArrayGatherShared<MDevT,T,IndexT>::contiguous_memory() const {
        return this->source.contiguous_memory() && this->indices.contiguous_memory();
    }

    template<typename MDevT, typename T, typename IndexT>
    bool TypedArrayGatherShared<MDevT,T,IndexT>::spans_entire_memory() const {
        return this->source.spans_entire_memory() && this->indices.spans_entire_memory();
    }

    template<typename MDevT, typename T, typename IndexT>
    mshadow::expr::TakeExp<mshadow::Tensor<MDevT, 1, IndexT>, mshadow::Tensor<MDevT, 1, T>, T, IndexT> TypedArrayGatherShared<MDevT,T, IndexT>::contiguous_d1(memory::AM access_mode, bool collapse_leading) const { return contiguous_d<1>(access_mode, collapse_leading); }
    template<typename MDevT, typename T, typename IndexT>
    mshadow::expr::TakeExp<mshadow::Tensor<MDevT, 1, IndexT>, mshadow::Tensor<MDevT, 2, T>, T, IndexT> TypedArrayGatherShared<MDevT,T, IndexT>::contiguous_d2(memory::AM access_mode, bool collapse_leading) const { return contiguous_d<2>(access_mode, collapse_leading); }
    template<typename MDevT, typename T, typename IndexT>
    mshadow::expr::TakeExp<mshadow::Tensor<MDevT, 1, IndexT>, mshadow::Tensor<MDevT, 3, T>, T, IndexT> TypedArrayGatherShared<MDevT,T, IndexT>::contiguous_d3(memory::AM access_mode, bool collapse_leading) const { return contiguous_d<3>(access_mode, collapse_leading); }
    template<typename MDevT, typename T, typename IndexT>
    mshadow::expr::TakeExp<mshadow::Tensor<MDevT, 1, IndexT>, mshadow::Tensor<MDevT, 4, T>, T, IndexT> TypedArrayGatherShared<MDevT,T, IndexT>::contiguous_d4(memory::AM access_mode, bool collapse_leading) const { return contiguous_d<4>(access_mode, collapse_leading); }

    template<typename MDevT, typename T, typename IndexT>
    mshadow::expr::TakeExp<DaliWrapperExp<MDevT, 1, IndexT>, DaliWrapperExp<MDevT, 1, T>, T, IndexT> TypedArrayGatherShared<MDevT,T, IndexT>::d1(memory::AM access_mode, bool collapse_leading) const { return d<1>(access_mode, collapse_leading); }
    template<typename MDevT, typename T, typename IndexT>
    mshadow::expr::TakeExp<DaliWrapperExp<MDevT, 1, IndexT>, DaliWrapperExp<MDevT, 2, T>, T, IndexT> TypedArrayGatherShared<MDevT,T, IndexT>::d2(memory::AM access_mode, bool collapse_leading) const { return d<2>(access_mode, collapse_leading); }
    template<typename MDevT, typename T, typename IndexT>
    mshadow::expr::TakeExp<DaliWrapperExp<MDevT, 1, IndexT>, DaliWrapperExp<MDevT, 3, T>, T, IndexT> TypedArrayGatherShared<MDevT,T, IndexT>::d3(memory::AM access_mode, bool collapse_leading) const { return d<3>(access_mode, collapse_leading); }
    template<typename MDevT, typename T, typename IndexT>
    mshadow::expr::TakeExp<DaliWrapperExp<MDevT, 1, IndexT>, DaliWrapperExp<MDevT, 4, T>, T, IndexT> TypedArrayGatherShared<MDevT,T, IndexT>::d4(memory::AM access_mode, bool collapse_leading) const { return d<4>(access_mode, collapse_leading); }

    template class TypedArrayGatherShared<mshadow::cpu, int, int>;
    template class TypedArrayGatherShared<mshadow::cpu, float, int>;
    template class TypedArrayGatherShared<mshadow::cpu, double, int>;
    #ifdef DALI_USE_CUDA
        template class TypedArrayGatherShared<mshadow::gpu, int, int>;
        template class TypedArrayGatherShared<mshadow::gpu, float, int>;
        template class TypedArrayGatherShared<mshadow::gpu, double, int>;
    #endif
} // namespace internal

template class TypedArray<memory::DEVICE_T_CPU, int>;
template class TypedArray<memory::DEVICE_T_CPU, float>;
template class TypedArray<memory::DEVICE_T_CPU, double>;

template class TypedArraySubtensor<memory::DEVICE_T_CPU, int, int>;
template class TypedArraySubtensor<memory::DEVICE_T_CPU, float, int>;
template class TypedArraySubtensor<memory::DEVICE_T_CPU, double, int>;

template class TypedArrayGather<memory::DEVICE_T_CPU, int, int>;
template class TypedArrayGather<memory::DEVICE_T_CPU, float, int>;
template class TypedArrayGather<memory::DEVICE_T_CPU, double, int>;


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

    template class TypedArrayGather<memory::DEVICE_T_GPU, int, int>;
    template class TypedArrayGather<memory::DEVICE_T_GPU, float, int>;
    template class TypedArrayGather<memory::DEVICE_T_GPU, double, int>;
#endif
