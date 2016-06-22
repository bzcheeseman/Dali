#include "conv.h"

using std::vector;

ConvLayer::ConvLayer(int out_channels_,
                     int in_channels_,
                     int filter_h_,
                     int filter_w_,
                     int stride_h_,
                     int stride_w_,
                     PADDING_T padding_,
                     const std::string& data_format_,
                     DType dtype_,
                     memory::Device device_) :
        out_channels(out_channels_),
        in_channels(in_channels_),
        filter_h(filter_h_),
        filter_w(filter_w_),
        stride_h(stride_h_),
        stride_w(stride_w_),
        padding(padding_),
        data_format(data_format_),
        AbstractLayer(dtype_, device_) {
    int funnel_in = filter_h * filter_w * in_channels;

    vector<int> W_shape;
    if (data_format == "NCHW") {
        W_shape = {out_channels, in_channels, filter_h, filter_w};
    } else {
        W_shape = {out_channels, filter_h, filter_w, in_channels};
    }
    W = Tensor::uniform(1.0 / std::sqrt(funnel_in), W_shape, dtype, device);
    b = Tensor::zeros({out_channels}, dtype, device);

}

ConvLayer::ConvLayer() {
}

ConvLayer::ConvLayer(const ConvLayer& other, bool copy_w, bool copy_dw) :
        out_channels(other.out_channels),
        in_channels(other.in_channels),
        filter_h(other.filter_h),
        filter_w(other.filter_w),
        stride_h(other.stride_h),
        stride_w(other.stride_w),
        padding(other.padding),
        data_format(other.data_format),
        AbstractLayer(other.dtype, other.device) {
    this->W = Tensor(other.W, copy_w, copy_dw);
    this->b = Tensor(other.b, copy_w, copy_dw);
}

Tensor ConvLayer::activate(Tensor in) const {
    auto res = tensor_ops::conv2d(in, W, stride_h, stride_w, padding, data_format);

    return tensor_ops::conv2d_add_bias(res, b, data_format);
}

ConvLayer ConvLayer::shallow_copy() const {
    return ConvLayer(*this, false, true);
}

std::vector<Tensor> ConvLayer::parameters() const {
    return {W, b};
}
