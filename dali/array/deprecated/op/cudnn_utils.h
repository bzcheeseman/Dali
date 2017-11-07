// #ifndef DALI_ARRAY_OP_CUDNN_UTILS_H
// #define DALI_ARRAY_OP_CUDNN_UTILS_H

// #include "dali/config.h"

// #ifdef DALI_USE_CUDNN

// #include <cudnn.h>
// #include <string>
// #include <vector>

// #include "dali/array/function/typed_array.h"
// #include "dali/array/expression/operator.h"
// #include "dali/array/memory/device.h"
// #include "dali/array/op/spatial/spatial_enums.h"


// namespace cudnn {
//     namespace wrapper {
//         template<typename Descriptor>
//         struct BaseTensor {
//             cudnnTensorFormat_t tensor_format;
//             cudnnDataType_t     dtype;
//             int n, c, h, w;

//             void* data;
//             Descriptor description;

//             template<typename T, int devT>
//             BaseTensor(TypedArray<devT,T> tensor,
//                        std::string data_format,
//                        memory::AM access_mode=memory::AM_READONLY);

//             ~BaseTensor();

//             operator std::string() const;

//         };


//         struct Tensor : public BaseTensor<cudnnTensorDescriptor_t> {
//             using BaseTensor<cudnnTensorDescriptor_t>::BaseTensor;
//         };

//         struct Filters : public BaseTensor<cudnnFilterDescriptor_t> {
//             using BaseTensor<cudnnFilterDescriptor_t>::BaseTensor;
//         };

//         struct Convolution {
//             const int padding_h;
//             const int padding_w;
//             const int stride_h;
//             const int stride_w;

//             cudnnConvolutionDescriptor_t description;

//             Convolution(int padding_h, int padding_w,
//                         int stride_h, int stride_w);

//             ~Convolution();

//             operator std::string() const;
//         };

//         struct Pooling {
//             const int window_h;
//             const int window_w;
//             const int padding_h;
//             const int padding_w;
//             const int stride_h;
//             const int stride_w;
//             cudnnPoolingMode_t pooling_mode;

//             cudnnPoolingDescriptor_t description;

//             Pooling(int window_h,  int window_w,
//                     int padding_h, int padding_w,
//                     int stride_h,  int stride_w,
//                     POOLING_T pooling_mode);

//             ~Pooling();

//             operator std::string() const;
//         };

//         struct Operator {
//           private:
//             float alpha_f;
//             float beta_f;
//             double alpha_d;
//             double beta_d;
//           public:
//             void* alpha_ptr;
//             void* beta_ptr;

//             Operator(OPERATOR_T operator_type, DType dtype);
//         };

//         std::ostream& operator<<(std::ostream& stream, const cudnn::wrapper::Tensor& info);
//         std::ostream& operator<<(std::ostream& stream, const cudnn::wrapper::Filters& info);
//         std::ostream& operator<<(std::ostream& stream, const cudnn::wrapper::Convolution& info);
//         std::ostream& operator<<(std::ostream& stream, const cudnn::wrapper::Pooling& info);
//     }

//     void conv2d(std::shared_ptr<wrapper::Tensor>  out,
//                 std::shared_ptr<wrapper::Tensor>  in,
//                 std::shared_ptr<wrapper::Filters> filters,
//                 std::shared_ptr<wrapper::Convolution> conv,
//                 const wrapper::Operator& update_operator);

//     void conv2d_bwd_input(std::shared_ptr<wrapper::Tensor>  in_dw,
//                           std::shared_ptr<wrapper::Filters> filters,
//                           std::shared_ptr<wrapper::Tensor>  out_dw,
//                           std::shared_ptr<wrapper::Convolution> conv,
//                           const wrapper::Operator& update_operator);

//     void conv2d_bwd_filters(std::shared_ptr<wrapper::Filters> filters_dw,
//                             std::shared_ptr<wrapper::Tensor>  input,
//                             std::shared_ptr<wrapper::Tensor>  out_dw,
//                             std::shared_ptr<wrapper::Convolution> conv,
//                             const wrapper::Operator& update_operator);


//     void conv2d_bwd_bias(std::shared_ptr<wrapper::Tensor> bias_dw,
//                          std::shared_ptr<wrapper::Tensor> out_dw,
//                          const wrapper::Operator& update_operator);

//     void pool2d(std::shared_ptr<wrapper::Tensor> out,
//                 std::shared_ptr<wrapper::Tensor>  in,
//                 std::shared_ptr<wrapper::Pooling> pooling,
//                 const wrapper::Operator& update_operator);

//     void pool2d_bwd(std::shared_ptr<wrapper::Tensor> in_dw,
//                     std::shared_ptr<wrapper::Tensor> out,
//                     std::shared_ptr<wrapper::Tensor> out_dw,
//                     std::shared_ptr<wrapper::Tensor> in,
//                     std::shared_ptr<wrapper::Pooling> pooling,
//                     const wrapper::Operator& update_operator);

// }  // namespace cudnn

// #endif  // DALI_USE_CUDNN
// #endif  // DALI_ARRAY_OP_CUDNN_UTILS_H
