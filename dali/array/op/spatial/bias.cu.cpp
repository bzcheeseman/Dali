#include "bias.h"
#include "dali/config.h"
#include "dali/runtime_config.h"
#include "dali/array/array.h"
#include "dali/array/function/function.h"

///////////////////////////////////////////////////////////////////////////////
//                    Conv2dBwdBiasFunction                                  //
///////////////////////////////////////////////////////////////////////////////

struct Conv2dBwdBiasFunction : public Function<Conv2dBwdBiasFunction,
                                        Array,
                                        Array,
                                        std::string> {
    static std::vector<int> deduce_output_bshape(const Array& out_dw,
                                                 const std::string& data_format) {
        size_t channel_dim = data_format.find('C');
        ASSERT2(channel_dim != std::string::npos,
                utils::MS() << "data_format must be NCHW or NHWC (got " << data_format << ").");

        return {out_dw.shape()[channel_dim]};
    }

    template<OPERATOR_T operator_t, typename T>
    void typed_eval(TypedArray<memory::DEVICE_T_CPU, T> bias_dw,
                    TypedArray<memory::DEVICE_T_CPU, T> out_dw,
                    const std::string& data_format) {
        throw std::runtime_error("not implemented!");
    }

#ifdef DALI_USE_CUDA
    template<OPERATOR_T operator_t, typename T, DALI_FUNC_ENABLE_IF_INT>
    void typed_eval(TypedArray<memory::DEVICE_T_GPU, T> bias_dw,
                    TypedArray<memory::DEVICE_T_GPU, T> out_dw,
                    const std::string& data_format) {
        ASSERT2(false, "integer Conv2dBwdBias is not implemented for GPU.");
    }

    template<OPERATOR_T operator_t, typename T, DALI_FUNC_DISABLE_IF_INT>
    void typed_eval(TypedArray<memory::DEVICE_T_GPU, T> bias_dw,
                    TypedArray<memory::DEVICE_T_GPU, T> out_dw,
                    const std::string& data_format) {

        auto out_access_mode = internal::OperatorAM<operator_t>::get(bias_dw);

        cudnn::conv2d_bwd_bias(
            std::make_shared<cudnn::wrapper::Tensor>(bias_dw, data_format, out_access_mode),
            std::make_shared<cudnn::wrapper::Tensor>(out_dw,  data_format),
            cudnn::wrapper::Operator(operator_t, template_to_dtype<T>())
        );
    }

#endif
};

namespace op {
    Assignable<Array> conv2d_backward_bias(
                     const Array& out_dw,
                     const std::string& data_format) {
        return Conv2dBwdBiasFunction::run(out_dw,data_format);
    }
}  // namespace op
