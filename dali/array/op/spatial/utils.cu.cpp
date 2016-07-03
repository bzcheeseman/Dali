#include "utils.h"
#include "dali/config.h"
#include "dali/runtime_config.h"
#include "dali/utils/assert2.h"
#include "dali/array/lazy/im2col.h"
#include "dali/utils/print_utils.h"

namespace internal {
    DataFormatDimMapping::DataFormatDimMapping(const std::string& data_format) {
        n_dim = data_format.find('N');
        c_dim = data_format.find('C');
        h_dim = data_format.find('H');
        w_dim = data_format.find('W');

        const auto MISS = std::string::npos;
        ASSERT2(n_dim != MISS && c_dim != MISS &&
                h_dim != MISS && w_dim != MISS &&
                data_format.size() == 4,
                utils::MS() << "data_format must be a permutation of letters N,C,H,W (was "
                            << data_format << ").");
    }

    template<typename Container>
    Container function_info_helper(
                const std::vector<int>& input_shape,
                const int& window_h,
                const int& window_w,
                const int& stride_h,
                const int& stride_w,
                const PADDING_T& padding,
                const DataFormatDimMapping& mapping) {
        Container info;

        info.batch_size         = input_shape[mapping.n_dim];
        info.in_channels        = input_shape[mapping.c_dim];
        info.in_h               = input_shape[mapping.h_dim];
        info.in_w               = input_shape[mapping.w_dim];

        if (padding == PADDING_T_SAME) {
            info.out_h = internal::int_ceil(info.in_h, stride_h);
            info.out_w = internal::int_ceil(info.in_w, stride_w);
        } else if (padding == PADDING_T_VALID) {
            info.out_h = internal::int_ceil(info.in_h - window_h + 1, stride_h);
            info.out_w = internal::int_ceil(info.in_w - window_w + 1, stride_w);
        } else {
            ASSERT2(false, utils::MS() << "Unrecognized value of padding passed to Pool2dFunction (got " << padding << ").");
        }

        if (padding == PADDING_T_SAME) {
            info.padding_h = std::max(0, (info.out_h - 1) * stride_h + window_h - info.in_h);
            info.padding_w = std::max(0, (info.out_w - 1) * stride_w + window_w - info.in_w);
        } else if (padding == PADDING_T_VALID) {
            info.padding_h = 0;
            info.padding_w = 0;
        }

        info.odd_padding_h = info.padding_h % 2;
        info.odd_padding_w = info.padding_w % 2;

        info.padding_h /= 2;
        info.padding_w /= 2;

        return info;
    }


    PoolFunctionInfo compute_pool_info(
            const std::vector<int>& input_shape,
            const int& window_h,
            const int& window_w,
            const int& stride_h,
            const int& stride_w,
            const PADDING_T& padding,
            const std::string& data_format) {
        internal::check_data_format(data_format);
        DataFormatDimMapping mapping(data_format);
        return function_info_helper<PoolFunctionInfo>(
                input_shape, window_h, window_w, stride_h, stride_w, padding, mapping);
    }

    ConvFunctionInfo compute_conv_info(
            const std::vector<int>& input_shape,
            const std::vector<int>& filters_shape,
            const int& stride_h,
            const int& stride_w,
            const PADDING_T& padding,
            const std::string& data_format) {
        internal::check_data_format(data_format);
        DataFormatDimMapping mapping(data_format);

        int filter_h = filters_shape[mapping.h_dim];
        int filter_w = filters_shape[mapping.w_dim];

        ConvFunctionInfo ret = function_info_helper<ConvFunctionInfo>(
                input_shape, filter_h, filter_w, stride_h, stride_w, padding, mapping);

        ASSERT2_EQ(ret.in_channels, filters_shape[mapping.c_dim],
            "Conv2d input and filters need to have the same number of input channels"
        );

        ret.filter_h     = filter_h;
        ret.filter_w     = filter_w;
        ret.out_channels = filters_shape[mapping.n_dim];
        return ret;
    }

    template<typename T, int devT>
    TypedArray<devT, T> compute_im2col(const TypedArray<devT, T>& input,
                                       const std::vector<int>& filter_shape,
                                       const int& stride_h,
                                       const int& stride_w,
                                       PADDING_T padding,
                                       const std::string& data_format) {
        auto info = compute_conv_info(input.array.shape(),
                                      filter_shape,
                                      stride_h,
                                      stride_w,
                                      padding,
                                      data_format);

        std::vector<int> temp_bshape;
        if (data_format == "NCHW") {
            temp_bshape = deduce_im2col_shape<mshadow::expr::DATA_FORMAT_NCHW>(
                input.array.shape(),
                info.filter_h, info.filter_w,
                stride_h, stride_w,
                /*dilate_h=*/1,
                /*dilate_w=*/1,
                /*padding_h*/2 * info.padding_h + info.odd_padding_h,
                /*padding_w*/2 * info.padding_w + info.odd_padding_w
            );
        } else {
            // when data_format is equal to the string containing
            // letters NHWC.
            temp_bshape = deduce_im2col_shape<mshadow::expr::DATA_FORMAT_NHWC>(
                input.array.shape(),
                info.filter_h, info.filter_w,
                stride_h, stride_w,
                /*dilate_h=*/1,
                /*dilate_w=*/1,
                /*padding_h*/2 * info.padding_h + info.odd_padding_h,
                /*padding_w*/2 * info.padding_w + info.odd_padding_w
            );
        }

        Array im2col_storage_arr(temp_bshape, template_to_dtype<T>(), input.device);
        TypedArray<devT, T> im2col_storage(im2col_storage_arr, input.device, temp_bshape);

        if (data_format == "NCHW") {
            im2col_storage.contiguous_d2(memory::AM_OVERWRITE) =
                    mshadow::expr::unpack_patch2col<mshadow::expr::DATA_FORMAT_NCHW>(
                input.d4(),
                info.filter_h,
                info.filter_w,
                stride_h,
                stride_w,
                /*dilate_h=*/1,
                /*dilate_w=*/1,
                /*prepad_h=*/info.padding_h,
                /*prepad_w=*/info.padding_w,
                /*postpad_h=*/info.padding_h + info.odd_padding_h,
                /*postpad_w=*/info.padding_w + info.odd_padding_w
            );
        } else { // then data_format = "NHWC"
            im2col_storage.contiguous_d2(memory::AM_OVERWRITE) =
                    mshadow::expr::unpack_patch2col<mshadow::expr::DATA_FORMAT_NHWC>(
                input.d4(),
                info.filter_h,
                info.filter_w,
                stride_h,
                stride_w,
                /*dilate_h=*/1,
                /*dilate_w=*/1,
                /*prepad_h=*/info.padding_h,
                /*prepad_w=*/info.padding_w,
                /*postpad_h=*/info.padding_h + info.odd_padding_h,
                /*postpad_w=*/info.padding_w + info.odd_padding_w
            );
        }
        return im2col_storage;
    }

    template TypedArray<memory::DEVICE_T_CPU, float> compute_im2col(const TypedArray<memory::DEVICE_T_CPU, float>&, const std::vector<int>&, const int&, const int&, PADDING_T, const std::string&);
    template TypedArray<memory::DEVICE_T_CPU, double> compute_im2col(const TypedArray<memory::DEVICE_T_CPU, double>&, const std::vector<int>&, const int&, const int&, PADDING_T, const std::string&);
    template TypedArray<memory::DEVICE_T_CPU, int> compute_im2col(const TypedArray<memory::DEVICE_T_CPU, int>&, const std::vector<int>&, const int&, const int&, PADDING_T, const std::string&);
#ifdef DALI_USE_CUDA
    template TypedArray<memory::DEVICE_T_GPU, float> compute_im2col(const TypedArray<memory::DEVICE_T_GPU, float>&, const std::vector<int>&, const int&, const int&, PADDING_T, const std::string&);
    template TypedArray<memory::DEVICE_T_GPU, double> compute_im2col(const TypedArray<memory::DEVICE_T_GPU, double>&, const std::vector<int>&, const int&, const int&, PADDING_T, const std::string&);
    template TypedArray<memory::DEVICE_T_GPU, int> compute_im2col(const TypedArray<memory::DEVICE_T_GPU, int>&, const std::vector<int>&, const int&, const int&, PADDING_T, const std::string&);
#endif

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
