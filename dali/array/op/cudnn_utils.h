#ifndef DALI_ARRAY_OP_CUDNN_UTILS_H
#define DALI_ARRAY_OP_CUDNN_UTILS_H

#include <cudnn.h>
#include <string>
#include <vector>

#include "dali/array/function/typed_array.h"
#include "dali/array/function/operator.h"
#include "dali/array/memory/device.h"


namespace cudnn {
    namespace wrapper {
        template<typename Descriptor>
        struct BaseTensor {
            void* data;
            Descriptor description;

            template<typename T>
            BaseTensor(TypedArray<memory::DEVICE_T_GPU,T> tensor,
                             std::string data_format,
                             memory::AM access_mode=memory::AM_READONLY);

            ~BaseTensor();
        };


        struct Tensor : public BaseTensor<cudnnTensorDescriptor_t> {
            using BaseTensor<cudnnTensorDescriptor_t>::BaseTensor;
        };

        struct Filters : public BaseTensor<cudnnFilterDescriptor_t> {
            using BaseTensor<cudnnFilterDescriptor_t>::BaseTensor;
        };

        struct Convolution {
            cudnnConvolutionDescriptor_t description;

            Convolution(int padding_h, int padding_w, int stride_h, int stride_w);

            ~Convolution();
        };

        struct Operator {
          private:
            float alpha_f;
            float beta_f;
            double alpha_d;
            double beta_d;
          public:
            void* alpha_ptr;
            void* beta_ptr;

            Operator(OPERATOR_T operator_type, DType dtype);
        };
    }

    void cudnn_conv2d(std::shared_ptr<wrapper::Tensor>  out,
                      std::shared_ptr<wrapper::Tensor>  in,
                      std::shared_ptr<wrapper::Filters> filters,
                      std::shared_ptr<wrapper::Convolution> conv,
                      const wrapper::Operator& update_operator);

    void cudnn_conv2d_bwd_input(std::shared_ptr<wrapper::Tensor>  in_dw,
                                std::shared_ptr<wrapper::Filters> filters,
                                std::shared_ptr<wrapper::Tensor>  out_dw,
                                std::shared_ptr<wrapper::Convolution> conv,
                                const wrapper::Operator& update_operator);
}  // namespace cudnn

#endif  // DALI_ARRAY_OP_CUDNN_UTILS_H
