#include "bias.h"

#include "dali/config.h"

#include "dali/array/array.h"
#include "dali/array/function/function.h"
#include "dali/array/op/cudnn_utils.h"
#include "dali/array/op/spatial/utils.h"
#include "dali/runtime_config.h"

///////////////////////////////////////////////////////////////////////////////
//                    Conv2dBwdBiasFunction                                  //
///////////////////////////////////////////////////////////////////////////////

struct Conv2dBwdBiasFunction : public Function<Conv2dBwdBiasFunction,
                                        Array,
                                        Array,
                                        std::string> {
    static std::vector<int> deduce_output_bshape(const Array& out_dw,
                                                 const std::string& data_format) {
        internal::DataFormatDimMapping mapping(data_format);
        ASSERT2_SHAPE_ND(out_dw.shape(), 4, "gradient for convolution bias");
        return {out_dw.bshape()[mapping.c_dim]};
    }


    template<OPERATOR_T operator_t, typename T, int devT>
    void typed_eval(TypedArray<devT, T> bias_dw,
                    TypedArray<devT, T> out_dw,
                    const std::string& data_format) {
#ifdef DALI_USE_CUDNN
        if (use_cudnn && devT == memory::DEVICE_T_GPU &&
                !std::is_same<T, int>::value &&
                operator_t != OPERATOR_T_MUL &&
                operator_t != OPERATOR_T_DIV &&
                out_dw.contiguous_memory()) {
            cudnn_conv_bias_bwd<operator_t,T,devT>(bias_dw, out_dw, data_format);
            return;
        }
#endif
        mshadow_conv_bias_bwd<operator_t,T,devT>(bias_dw, out_dw, data_format);
    }


    template<OPERATOR_T operator_t, typename T, int devT>
    void mshadow_conv_bias_bwd(TypedArray<devT, T> bias_dw,
                               TypedArray<devT, T> out_dw,
                               const std::string& data_format) {
        int c_channel = data_format.find('C');

        switch(c_channel) {
            case 0:
                operator_assign<operator_t, 1>(
                    bias_dw,
                    mshadow::expr::sumall_except_dim<0>(
                        out_dw.d4()
                    )
                );
                break;
            case 1:
                operator_assign<operator_t, 1>(
                    bias_dw,
                    mshadow::expr::sumall_except_dim<1>(
                        out_dw.d4()
                    )
                );
                break;
            case 2:
                operator_assign<operator_t, 1>(
                    bias_dw,
                    mshadow::expr::sumall_except_dim<2>(
                        out_dw.d4()
                    )
                );
                break;
            case 3:
                operator_assign<operator_t, 1>(
                    bias_dw,
                    mshadow::expr::sumall_except_dim<3>(
                        out_dw.d4()
                    )
                );
                break;
            default:
                ASSERT2(false, utils::MS() << "data_format not understood: " << data_format);
                break;
        }
    }

#ifdef DALI_USE_CUDNN
    template<OPERATOR_T operator_t, typename T, int devT>
    void cudnn_conv_bias_bwd(TypedArray<devT, T> bias_dw,
                             TypedArray<devT, T> out_dw,
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
