#ifndef DALI_ARRAY_FUNCTION_TYPED_ARRAY_H
#define DALI_ARRAY_FUNCTION_TYPED_ARRAY_H

#include "dali/config.h"
#include <mshadow/tensor.h>
#include <vector>
#ifdef DALI_USE_CUDA
    #include <thrust/device_vector.h>
#endif

#include "dali/array/array.h"
#include "dali/array/function/args/dali_wrapper_exp.h"

////////////////////////////////////////////////////////////////////////////////
//                                   UTILS                                    //
////////////////////////////////////////////////////////////////////////////////

namespace internal {
    template<int dstdim>
    mshadow::Shape<dstdim> canonical_reshape(const std::vector<int>& src_shape);
}
////////////////////////////////////////////////////////////////////////////////
//                            TYPED ARRAY SHARED                              //
//                                   ---                                      //
//  Common to both CPU and GPU implementations of TypedArray below.           //
////////////////////////////////////////////////////////////////////////////////

namespace internal {
    template<typename MDevT, typename T>
    struct TypedArrayShared {
      private:
        template<int dim>
        mshadow::Tensor<MDevT, dim, T> mtensor(memory::AM access_mode=memory::AM_READONLY) const;

        T* ptr_internal(memory::AM access_mode=memory::AM_READONLY) const;
      public:
        mutable Array array;
        memory::Device device;
        std::vector<int> output_shape; 

        T* ptr(memory::AM access_mode=memory::AM_READONLY) const;

        template<int dim>
        mshadow::Tensor<MDevT, dim, T> contiguous_d(memory::AM access_mode=memory::AM_READONLY) const;

        template<int dim>
        DaliWrapperExp<MDevT, dim, T> d(memory::AM access_mode=memory::AM_READONLY) const;

        TypedArrayShared(const Array& _array, const memory::Device& _device, const std::vector<int>& output_shape);

        ///////////////////// CONVINENCE WARPPERS //////////////////////////////////
        mshadow::Tensor<MDevT, 1, T> contiguous_d1(memory::AM access_mode=memory::AM_READONLY) const;
        mshadow::Tensor<MDevT, 2, T> contiguous_d2(memory::AM access_mode=memory::AM_READONLY) const;
        mshadow::Tensor<MDevT, 3, T> contiguous_d3(memory::AM access_mode=memory::AM_READONLY) const;
        mshadow::Tensor<MDevT, 4, T> contiguous_d4(memory::AM access_mode=memory::AM_READONLY) const;

        DaliWrapperExp<MDevT, 1, T> d1(memory::AM access_mode=memory::AM_READONLY) const;
        DaliWrapperExp<MDevT, 2, T> d2(memory::AM access_mode=memory::AM_READONLY) const;
        DaliWrapperExp<MDevT, 3, T> d3(memory::AM access_mode=memory::AM_READONLY) const;
        DaliWrapperExp<MDevT, 4, T> d4(memory::AM access_mode=memory::AM_READONLY) const;
    };
}  // namespace internal

////////////////////////////////////////////////////////////////////////////////
//                               TYPED ARRAY                                  //
//                                   ---                                      //
//  Different code needs to execute based on type of data and device type     //
//  that it is running on. Typed array is a set of utilities which allow to   //
//  access the array in different data/device access modes.                   //
////////////////////////////////////////////////////////////////////////////////
template<int devT, typename T>
struct TypedArray {
    // some people use this but not sure who...
};

template<typename T>
struct TypedArray<memory::DEVICE_T_CPU, T> : public internal::TypedArrayShared<mshadow::cpu,T> {
    using internal::TypedArrayShared<mshadow::cpu,T>::TypedArrayShared; // inherit parent constructor
};

#ifdef DALI_USE_CUDA
    template<typename T>
    struct TypedArray<memory::DEVICE_T_GPU, T> : public internal::TypedArrayShared<mshadow::gpu,T> {
        using internal::TypedArrayShared<mshadow::gpu,T>::TypedArrayShared; // inherit parent constructor

        thrust::device_ptr<T> to_thrust(memory::AM access_mode=memory::AM_READONLY) const;
    };
#endif

#include "dali/array/function/typed_array-impl.h"

#endif // DALI_ARRAY_FUNCTION_TYPED_ARRAY_H
