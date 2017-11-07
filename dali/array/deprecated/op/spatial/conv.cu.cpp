// #include "conv.h"

// #include "dali/config.h"

// #include "dali/array/array.h"
// #include "dali/array/function/function.h"
// #include "dali/array/expression/operator.h"
// #include "dali/array/functor.h"
// #include "dali/array/lazy/im2col.h"
// #include "dali/array/mshadow_extension/dali_gemm_engine_exp.h"
// #include "dali/array/op/cudnn_utils.h"
// #include "dali/array/op/spatial/utils.h"
// #include "dali/runtime_config.h"
// #include "dali/utils/random.h"

// using internal::DataFormatDimMapping;

// ///////////////////////////////////////////////////////////////////////////////
// //                            Conv2dFunction                                 //
// ///////////////////////////////////////////////////////////////////////////////

// struct Conv2dFunction : public Function<Conv2dFunction,
//                                         Array,
//                                         Array,
//                                         Array,
//                                         int,
//                                         int,
//                                         PADDING_T,
//                                         std::string> {
//     static std::string name;

//     static std::vector<int> deduce_output_bshape(const Array& input,
//                                                  const Array& filters,
//                                                  int stride_h,
//                                                  int stride_w,
//                                                  PADDING_T padding,
//                                                  const std::string& data_format) {

//         auto info = internal::compute_conv_info(input.shape(),
//                                       filters.shape(),
//                                       stride_h,
//                                       stride_w,
//                                       padding,
//                                       data_format);

//         ASSERT2_SHAPE_ND(input.shape(),   4, "Conv2dFunction input");
//         ASSERT2_SHAPE_ND(filters.shape(), 4, "Conv2dFunction filters");

//         if (data_format == "NCHW") {
//             return std::vector<int> {info.batch_size, info.out_channels, info.out_h, info.out_w};
//         } else { // then data_format == "NHWC":
//             return std::vector<int> {info.batch_size, info.out_h, info.out_w, info.out_channels};
//         }
//     }

//     template<OPERATOR_T operator_t, typename T, int devT, DALI_FUNC_ENABLE_IF_MUL_DIV>
//     void typed_eval(TypedArray<devT, T> out,
//                     TypedArray<devT, T> input,
//                     TypedArray<devT, T> filters,
//                     int stride_h,
//                     int stride_w,
//                     PADDING_T padding,
//                     const std::string& data_format) {
//         ASSERT2(!(var_operator_t == OPERATOR_T_MUL) && !(var_operator_t == OPERATOR_T_MUL),
//                 "Convolution's result cannot be inplace-multiplied or inplace-divided.");
//         ASSERT2(false, "If asserts above are complete this message should never be displayed");
//     }

//     template<OPERATOR_T operator_t, typename T, int devT, DALI_FUNC_DISABLE_IF_MUL_DIV>
//     void typed_eval(TypedArray<devT, T> out,
//                     TypedArray<devT, T> input,
//                     TypedArray<devT, T> filters,
//                     int stride_h,
//                     int stride_w,
//                     PADDING_T padding,
//                     const std::string& data_format) {
// #ifdef DALI_USE_CUDNN
//         if (use_cudnn && devT == memory::DEVICE_T_GPU &&
//                 !std::is_same<T, int>::value &&
//                 !(data_format == "NHWC" && std::is_same<T, double>::value)) {
//             cudnn_conv<operator_t,T,devT>(out, input, filters, stride_h, stride_w, padding, data_format);
//             return;
//         }
// #endif
//         blas_conv<operator_t,T,devT>(out, input, filters, stride_h, stride_w, padding, data_format);
//     }

//     template<OPERATOR_T operator_t, typename T, int devT, DALI_FUNC_DISABLE_IF_MUL_DIV>
//     void blas_conv(TypedArray<devT, T> out,
//                    TypedArray<devT, T> input,
//                    TypedArray<devT, T> filters,
//                    int stride_h,
//                    int stride_w,
//                    PADDING_T padding,
//                    const std::string& data_format) {
//         auto im2col_storage = internal::compute_im2col(
//             input, filters.array.shape(), stride_h, stride_w, padding, data_format
//         );

//         typedef decltype(im2col_storage.contiguous_d2()) mshadow_tensor_t;

//         filters.array   = filters.array.copyless_reshape({filters.array.shape()[0], -1});
//         if (data_format == "NCHW") {
//             // do nothing
//         } else {
//             im2col_storage.array = im2col_storage.array.transpose();
//             filters.array        = filters.array.transpose();
//         }

//         bool             im2col_transposed, filters_transposed;
//         mshadow_tensor_t im2col_tensor,     filters_tensor;
//         std::tie(im2col_transposed,   im2col_tensor)  = im2col_storage.blas_friendly_tensor();
//         std::tie(filters_transposed, filters_tensor) = filters.blas_friendly_tensor();

//         if (data_format == "NCHW") {
//             auto out_cnhw_shape = out.array.shape();
//             std::swap(out_cnhw_shape[0], out_cnhw_shape[1]);
//             Array out_cnhw_arr(out_cnhw_shape, template_to_dtype<T>(), out.device);
//             TypedArray<devT, T> out_cnhw(out_cnhw_arr, input.device, out_cnhw_shape);

//             operator_assign_contiguous<OPERATOR_T_EQL, 2>(
//                 out_cnhw,
//                 dali_gemm(
//                     filters_tensor,
//                     im2col_tensor,
//                     filters_transposed,
//                     im2col_transposed,
//                     (T)1.0f
//                 ),
//                 /*collapse_leading=*/false
//             );
//             operator_assign_contiguous<operator_t, 4>(
//                 out,
//                 mshadow::expr::swapaxis<1,0>(out_cnhw.contiguous_d4())
//             );
//         } else {
//             auto out_2d_arr = out.array.copyless_reshape({-1, out.array.shape()[3]});
//             TypedArray<devT, T> out_2d(out_2d_arr, out.device, out_2d_arr.shape());

//             operator_assign_contiguous<operator_t, 2>(
//                 out_2d,
//                 dali_gemm(
//                     im2col_tensor,
//                     filters_tensor,
//                     im2col_transposed,
//                     filters_transposed,
//                     (T)1.0f
//                 )
//             );
//         }
//     }

// #ifdef DALI_USE_CUDNN
//     template<OPERATOR_T operator_t, typename T, int devT, DALI_FUNC_DISABLE_IF_MUL_DIV>
//     void cudnn_conv(TypedArray<devT, T> out,
//                     TypedArray<devT, T> input,
//                     TypedArray<devT, T> filters,
//                     int stride_h,
//                     int stride_w,
//                     PADDING_T padding,
//                     const std::string& data_format) {
//         auto info = internal::compute_conv_info(input.array.shape(),
//                                                 filters.array.shape(),
//                                                 stride_h,
//                                                 stride_w,
//                                                 padding,
//                                                 data_format);
//         TypedArray<devT, T> maybe_copied_input = input;

//         // This whole shenanigans is needed because
//         // cudnn does not support odd padding.
//         // If it is any consolation TF does it as well.
//         if (info.odd_padding_h || info.odd_padding_w) {
//             // compute padded shape
//             DataFormatDimMapping mapping(data_format);

//             auto padded_shape = input.array.shape();
//             if (info.odd_padding_h) padded_shape[mapping.h_dim] += 1;
//             if (info.odd_padding_w) padded_shape[mapping.w_dim] += 1;

//             // create temporary storage for padded array.
//             auto padded_input_arr = Array::zeros(padded_shape,
//                                                  input.array.dtype(),
//                                                  input.device);
//             TypedArray<devT,T> padded_input(padded_input_arr, input.device, padded_shape);
//             maybe_copied_input = padded_input;

//             // copy values from source array over
//             Array padded_input_slice_arr = padded_input_arr;
//             if (info.odd_padding_h) {
//                 padded_input_slice_arr = padded_input_slice_arr.pluck_axis(
//                         mapping.h_dim, Slice(0, padded_shape[mapping.h_dim] -1));
//             }
//             if (info.odd_padding_w) {
//                 padded_input_slice_arr = padded_input_slice_arr.pluck_axis(
//                         mapping.w_dim, Slice(0, padded_shape[mapping.w_dim] -1));
//             }

//             TypedArray<devT,T> padded_input_slice(padded_input_slice_arr,
//                                                   padded_input.device,
//                                                   input.array.shape());
//             padded_input_slice.d2(memory::AM_MUTABLE) =
//                     mshadow::expr::F<mshadow::op::identity>(input.d2());
//         }

//         auto out_access_mode = internal::OperatorAM<operator_t>::get(out);

//         cudnn::conv2d(
//                 std::make_shared<cudnn::wrapper::Tensor>(
//                         out, data_format, out_access_mode),
//                 std::make_shared<cudnn::wrapper::Tensor>(
//                         maybe_copied_input, data_format),
//                 std::make_shared<cudnn::wrapper::Filters>(
//                         filters, data_format),
//                 std::make_shared<cudnn::wrapper::Convolution>(
//                         info.padding_h, info.padding_w, stride_h, stride_w),
//                 cudnn::wrapper::Operator(operator_t, template_to_dtype<T>())
//         );
//     }
// #endif
// };

// std::string Conv2dFunction::name = "conv2d";

// namespace old_op {
//     Assignable<Array> conv2d(const Array& input,
//                              const Array& filters,
//                              int stride_h,
//                              int stride_w,
//                              PADDING_T padding,
//                              const std::string& data_format) {
//         return Conv2dFunction::run(input,
//                                    filters,
//                                    stride_h,
//                                    stride_w,
//                                    padding,
//                                    data_format);
//     }

//     Assignable<Array> im2col(const Array& input,
//                              int filter_h,
//                              int filter_w,
//                              int stride_h,
//                              int stride_w,
//                              const std::string& data_format) {
//         internal::check_data_format(data_format);
//         if (data_format == "NCHW") {
//             return lazy::im2col_nchw(input, filter_h, filter_w, stride_h, stride_w, 1, 1);
//         } else {
//             return lazy::im2col_nhwc(input, filter_h, filter_w, stride_h, stride_w, 1, 1);
//         }
//     }

//     Assignable<Array> col2im(const Array& input,
//                              const std::vector<int>& image_shape,
//                              int filter_h,
//                              int filter_w,
//                              int stride_h,
//                              int stride_w,
//                              const std::string& data_format) {
//         internal::check_data_format(data_format);
//         if (data_format == "NCHW") {
//             return lazy::col2im_nchw(input, image_shape, filter_h, filter_w, stride_h, stride_w, 1, 1);
//         } else {
//             return lazy::col2im_nhwc(input, image_shape, filter_h, filter_w, stride_h, stride_w, 1, 1);
//         }
//     }
// }  // namespace old_op
