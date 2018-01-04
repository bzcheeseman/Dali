#include "utils.h"

#include "dali/config.h"

#include "dali/runtime_config.h"
#include "dali/utils/assert2.h"
// #include "dali/array/lazy/im2col.h"
#include "dali/utils/print_utils.h"

namespace internal {


//     template<typename T, int devT>
//     TypedArray<devT, T> compute_im2col(const TypedArray<devT, T>& input,
//                                        const std::vector<int>& filter_shape,
//                                        const int& stride_h,
//                                        const int& stride_w,
//                                        PADDING_T padding,
//                                        const std::string& data_format) {
//         auto info = compute_conv_info(input.array.shape(),
//                                       filter_shape,
//                                       stride_h,
//                                       stride_w,
//                                       padding,
//                                       data_format);

//         std::vector<int> temp_bshape;
//         if (data_format == "NCHW") {
//             temp_bshape = deduce_im2col_shape<mshadow::expr::DATA_FORMAT_NCHW>(
//                 input.array.shape(),
//                 info.filter_h, info.filter_w,
//                 stride_h, stride_w,
//                 dilate_h=1,
//                 /*dilate_w=*/1,
//                 /*padding_h*/2 * info.padding_h + info.odd_padding_h,
//                 /*padding_w*/2 * info.padding_w + info.odd_padding_w
//             );
//         } else { // then data_format = "NHWC"
//             temp_bshape = deduce_im2col_shape<mshadow::expr::DATA_FORMAT_NHWC>(
//                 input.array.shape(),
//                 info.filter_h, info.filter_w,
//                 stride_h, stride_w,
//                 /*dilate_h=*/1,
//                 /*dilate_w=*/1,
//                 /*padding_h*/2 * info.padding_h + info.odd_padding_h,
//                 /*padding_w*/2 * info.padding_w + info.odd_padding_w
//             );
//         }

//         Array im2col_storage_arr(temp_bshape, template_to_dtype<T>(), input.device);
//         TypedArray<devT, T> im2col_storage(im2col_storage_arr, input.device, temp_bshape);

//         if (data_format == "NCHW") {
//             im2col_storage.contiguous_d2(memory::AM_OVERWRITE) =
//                     mshadow::expr::unpack_patch2col<mshadow::expr::DATA_FORMAT_NCHW>(
//                 input.d4(),
//                 info.filter_h,
//                 info.filter_w,
//                 stride_h,
//                 stride_w,
//                 /*dilate_h=*/1,
//                 /*dilate_w=*/1,
//                 /*prepad_h=*/info.padding_h,
//                 /*prepad_w=*/info.padding_w,
//                 /*postpad_h=*/info.padding_h + info.odd_padding_h,
//                 /*postpad_w=*/info.padding_w + info.odd_padding_w
//             );
//         } else { // then data_format = "NHWC"
//             im2col_storage.contiguous_d2(memory::AM_OVERWRITE) =
//                     mshadow::expr::unpack_patch2col<mshadow::expr::DATA_FORMAT_NHWC>(
//                 input.d4(),
//                 info.filter_h,
//                 info.filter_w,
//                 stride_h,
//                 stride_w,
//                 /*dilate_h=*/1,
//                 /*dilate_w=*/1,
//                 /*prepad_h=*/info.padding_h,
//                 /*prepad_w=*/info.padding_w,
//                 /*postpad_h=*/info.padding_h + info.odd_padding_h,
//                 /*postpad_w=*/info.padding_w + info.odd_padding_w
//             );
//         }
//         return im2col_storage;
//     }

//     template TypedArray<memory::DEVICE_T_CPU, float> compute_im2col(const TypedArray<memory::DEVICE_T_CPU, float>&, const std::vector<int>&, const int&, const int&, PADDING_T, const std::string&);
//     template TypedArray<memory::DEVICE_T_CPU, double> compute_im2col(const TypedArray<memory::DEVICE_T_CPU, double>&, const std::vector<int>&, const int&, const int&, PADDING_T, const std::string&);
//     template TypedArray<memory::DEVICE_T_CPU, int> compute_im2col(const TypedArray<memory::DEVICE_T_CPU, int>&, const std::vector<int>&, const int&, const int&, PADDING_T, const std::string&);
// #ifdef DALI_USE_CUDA
//     template TypedArray<memory::DEVICE_T_GPU, float> compute_im2col(const TypedArray<memory::DEVICE_T_GPU, float>&, const std::vector<int>&, const int&, const int&, PADDING_T, const std::string&);
//     template TypedArray<memory::DEVICE_T_GPU, double> compute_im2col(const TypedArray<memory::DEVICE_T_GPU, double>&, const std::vector<int>&, const int&, const int&, PADDING_T, const std::string&);
//     template TypedArray<memory::DEVICE_T_GPU, int> compute_im2col(const TypedArray<memory::DEVICE_T_GPU, int>&, const std::vector<int>&, const int&, const int&, PADDING_T, const std::string&);
// #endif

}  // namespace internal

std::ostream& operator<<(std::ostream& stream, const internal::PoolFunctionInfo& info) {
  return stream << "PoolFunctionInfo(\n"
                << "  batch_size=" << info.batch_size << ",\n"
                << "  in_channels=" << info.in_channels << ",\n"
                << "  in_h=" << info.in_h << ",\n"
                << "  in_w=" << info.in_w << ",\n"
                << "  out_h=" << info.out_h << ",\n"
                << "  out_w=" << info.out_w << ",\n"
                << "  padding_h=" << info.padding_h << ",\n"
                << "  padding_w=" << info.padding_w << ",\n"
                << "  odd_padding_h=" << info.odd_padding_h << ",\n"
                << "  odd_padding_w=" << info.odd_padding_w << "\n"
                << ")";
}

std::ostream& operator<<(std::ostream& stream, const internal::ConvFunctionInfo& info) {
  return stream << "ConvFunctionInfo(\n"
                << "  batch_size=" << info.batch_size << ",\n"
                << "  in_channels=" << info.in_channels << ",\n"
                << "  in_h=" << info.in_h << ",\n"
                << "  in_w=" << info.in_w << ",\n"
                << "  filter_h=" << info.filter_h << ",\n"
                << "  filter_w=" << info.filter_w << ",\n"
                << "  out_channels=" << info.out_channels << ",\n"
                << "  out_h=" << info.out_h << ",\n"
                << "  out_w=" << info.out_w << ",\n"
                << "  padding_h=" << info.padding_h << ",\n"
                << "  padding_w=" << info.padding_w << ",\n"
                << "  odd_padding_h=" << info.odd_padding_h << ",\n"
                << "  odd_padding_w=" << info.odd_padding_w << "\n"
                << ")";
}
