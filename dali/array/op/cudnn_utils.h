#ifndef DALI_ARRAY_OP_CUDNN_UTILS_H
#define DALI_ARRAY_OP_CUDNN_UTILS_H

#include <cudnn.h>
#include <string>
#include <vector>

#include "dali/array/function/typed_array.h"
#include "dali/array/memory/device.h"


template<typename Descriptor>
struct DaliCudnnWrapper {
    void* data;
    Descriptor description;

    template<typename T>
    DaliCudnnWrapper(TypedArray<memory::DEVICE_T_GPU,T> tensor,
                     std::string data_format,
                     memory::AM access_mode=memory::AM_READONLY);

    ~DaliCudnnWrapper();
};


struct DaliCudnnTensor : public DaliCudnnWrapper<cudnnTensorDescriptor_t> {
    using DaliCudnnWrapper<cudnnTensorDescriptor_t>::DaliCudnnWrapper;
};

struct DaliCudnnFilters : public DaliCudnnWrapper<cudnnFilterDescriptor_t> {
    using DaliCudnnWrapper<cudnnFilterDescriptor_t>::DaliCudnnWrapper;
};

namespace cudnn_utils {
   void cudnn_conv2d(std::shared_ptr<DaliCudnnTensor>  out,
                     std::shared_ptr<DaliCudnnTensor>  in,
                     std::shared_ptr<DaliCudnnFilters> filters,
                     int stride_w,
                     int stride_h,
                     int padding_h,
                     int padding_w,
                     double alpha,
                     double beta,
                     DType dtype);
}

#endif  // DALI_ARRAY_OP_CUDNN_UTILS_H
