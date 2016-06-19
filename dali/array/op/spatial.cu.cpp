#include "spatial.h"

#include "dali/array/array.h"

#include "dali/array/function/function.h"
#include "dali/array/function/operator.h"
#include "dali/array/functor.h"
#include "dali/config.h"
#include "dali/utils/random.h"
#ifdef DALI_USE_CUDA
    #include "dali/array/op/cudnn_utils.h"
#endif

#define ASSERT2_SHAPE_ND(SHAPE,DIM,MSG) \
    if (SHAPE.size() != DIM) \
        utils::assert2(false, utils::MS() << MSG << " was expecting dimension " << DIM  << ", got shape " << SHAPE << ".");

#define ASSERT2_EQ(EXPECTED,ACTUAL,MSG) \
    if (EXPECTED != ACTUAL) \
        utils::assert2(false, utils::MS() << "Expected " << EXPECTED  << ", got " << ACTUAL << ": " << MSG << ".");


int int_ceil(int numerator, int denominator) {
    return (numerator + denominator - 1) / denominator;
}

std::tuple<double,double> operator_to_alphabeta(OPERATOR_T operator_t) {
    switch (operator_t) {
        case OPERATOR_T_EQL:
            return std::tuple<double, double>(1.0, 0.0);
        case OPERATOR_T_ADD:
            return std::tuple<double, double>(1.0, 1.0);
        case OPERATOR_T_SUB:
            return std::tuple<double, double>(-1.0, 1.0);
        default:
            ASSERT2(false, "This function only supports =,+,-");
            return std::tuple<double, double>(0.0, 0.0);
    }

}

struct Conv2dFunction : public Function<Conv2dFunction,
                                        Array,
                                        Array,
                                        Array,
                                        int,
                                        int,
                                        op::PADDING_T,
                                        std::string> {
    static std::vector<int> deduce_output_bshape(const Array& input,
                                                 const Array& filters,
                                                 int stride_h,
                                                 int stride_w,
                                                 op::PADDING_T padding,
                                                 std::string data_format) {

        ASSERT2_SHAPE_ND(input.shape(),   4, "Conv2dFunction input");
        ASSERT2_SHAPE_ND(filters.shape(), 4, "Conv2dFunction filters");

        ASSERT2(data_format == "NCHW" || data_format == "NHWC",
            utils::MS() << "data_format must be one of NCHW, NHWC (was " << data_format << ")");

        int out_channels, out_w, out_h;

        int batch_size, in_channels, in_h, in_w;
        int filter_in_channels, filter_h, filter_w;


        if (data_format == "NCHW") {
            batch_size         = input.shape()[0];
            in_channels        = input.shape()[1];
            in_h               = input.shape()[2];
            in_w               = input.shape()[3];

            out_channels       = filters.shape()[0];
            filter_in_channels = filters.shape()[1];
            filter_h           = filters.shape()[2];
            filter_w           = filters.shape()[3];

        } else if (data_format == "NHWC") {
            batch_size         = input.shape()[0];
            in_h               = input.shape()[1];
            in_w               = input.shape()[2];
            in_channels        = input.shape()[3];

            out_channels       = filters.shape()[0];
            filter_h           = filters.shape()[1];
            filter_w           = filters.shape()[2];
            filter_in_channels = filters.shape()[3];
        }

        ASSERT2_EQ(in_channels, filter_in_channels, "Conv2dFunction input and filters need to have the same number of input channels");

        if (padding == op::PADDING_T_SAME) {
            out_h = int_ceil(in_h, stride_h);
            out_w = int_ceil(in_w, stride_w);
        } else if (padding == op::PADDING_T_VALID) {
            out_h = int_ceil(in_h - filter_h + 1, stride_h);
            out_w = int_ceil(in_w - filter_w + 1, stride_w);
        } else {
            ASSERT2(false, utils::MS() << "Unrecognized value of padding passed to Conv2dFunction (" << padding << ")");
        }

        if (data_format == "NCHW") {
            return std::vector<int> {batch_size, out_channels, out_h, out_w};
        } else {
            return std::vector<int> {batch_size, out_h, out_w, out_channels};
        }
    }

#ifdef DALI_USE_CUDA
    template<OPERATOR_T operator_t>
    void typed_eval(TypedArray<memory::DEVICE_T_GPU, int> out,
                    TypedArray<memory::DEVICE_T_GPU, int> input,
                    TypedArray<memory::DEVICE_T_GPU, int> filters,
                    int stride_h,
                    int stride_w,
                    op::PADDING_T padding,
                    std::string data_format) {
        ASSERT2(false,
            "integer convolution is not implemented for GPU.");
    }

    template<OPERATOR_T operator_t, typename T>
    void typed_eval(TypedArray<memory::DEVICE_T_GPU, T> out,
                    TypedArray<memory::DEVICE_T_GPU, T> input,
                    TypedArray<memory::DEVICE_T_GPU, T> filters,
                    int stride_h,
                    int stride_w,
                    op::PADDING_T padding,
                    std::string data_format) {

        double alpha, beta;
        std::tie(alpha, beta) = operator_to_alphabeta(operator_t);

        int h_dim, w_dim;
        if (data_format == "NCHW") {
            h_dim = 2;
            w_dim = 3;
        } else if (data_format == "NHWC") {
            h_dim = 1;
            w_dim = 2;
        }

        int in_h     = input.array.shape()[h_dim];
        int in_w     = input.array.shape()[w_dim];
        int out_h    = out.array.shape()[h_dim];
        int out_w    = out.array.shape()[w_dim];
        int filter_h = filters.array.shape()[h_dim];
        int filter_w = filters.array.shape()[w_dim];;

        int padding_h, padding_w;


        if (padding == op::PADDING_T_SAME) {
            padding_h = (out_h - 1) * stride_h + filter_h - in_h;
            padding_w = (out_w - 1) * stride_w + filter_w - in_w;
            ASSERT2(padding_h % 2 == 0 && padding_w % 2 == 0,
                    "Conv2d odd sized padding is not supported at the moment");
            padding_h /= 2;
            padding_w /= 2;
        } else if (padding == op::PADDING_T_VALID) {
            padding_h = 0;
            padding_w = 0;
        }
        memory::AM out_access_mode;
        if (operator_t == OPERATOR_T_EQL) {
            out_access_mode = memory::AM_OVERWRITE;
        } else {
            out_access_mode = memory::AM_MUTABLE;
        }

        cudnn_utils::cudnn_conv2d(
                std::make_shared<DaliCudnnTensor>(out, data_format, out_access_mode),
                std::make_shared<DaliCudnnTensor>(input, data_format),
                std::make_shared<DaliCudnnFilters>(filters, data_format),
                stride_w,
                stride_h,
                padding_h,
                padding_w,
                alpha,
                beta,
                template_to_dtype<T>()
        );
    }

#endif

    template<OPERATOR_T operator_t, typename T>
    void typed_eval(TypedArray<memory::DEVICE_T_CPU, T> out,
                    TypedArray<memory::DEVICE_T_CPU, T> input,
                    TypedArray<memory::DEVICE_T_CPU, T> filters,
                    int stride_h,
                    int stride_w,
                    op::PADDING_T padding,
                    std::string data_format) {
        throw std::runtime_error("not implemented!");
    }
};




namespace op {
    Assignable<Array> conv2d(const Array& input,
                             const Array& filters,
                             int stride_h,
                             int stride_w,
                             PADDING_T padding,
                             std::string data_format) {
        return Conv2dFunction::run(input, filters, stride_h, stride_w, padding, data_format);
    }
};  // namespace op
