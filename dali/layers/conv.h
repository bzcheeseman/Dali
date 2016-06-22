#ifndef DALI_LAYERS_CONV_H
#define DALI_LAYERS_CONV_H

#include <initializer_list>

#include "dali/array/memory/device.h"
#include "dali/array/dtype.h"
#include "dali/runtime_config.h"
#include "dali/tensor/tensor.h"
#include "dali/tensor/op.h"
#include "dali/layers/layers.h"

class ConvLayer : public AbstractLayer {
  public:
    Tensor W;
    Tensor b;

    int out_channels;
    int in_channels;
    int filter_h;
    int filter_w;
    int stride_h;
    int stride_w;
    PADDING_T padding;
    std::string data_format;

    ConvLayer(int out_channels_,
              int in_channels_,
              int filter_h_,
              int filter_w_,
              int stride_h_=1,
              int stride_w_=1,
              PADDING_T padding_=PADDING_T_SAME,
              const std::string& data_format_="NCHW",
              DType dtype_=DTYPE_FLOAT,
              memory::Device device_=memory::default_preferred_device);

    ConvLayer();

    ConvLayer(const ConvLayer& other, bool copy_w, bool copy_dw);

    Tensor activate(Tensor in) const;

    ConvLayer shallow_copy() const;

    virtual std::vector<Tensor> parameters() const;
};


#endif  // DALI_LAYERS_CONV_H
