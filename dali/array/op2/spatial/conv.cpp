#include "conv.h"

#include "dali/config.h"
#include "dali/array/op2/expression/expression.h"
#include "dali/array/op2/expression/array_wrapper.h"
#include "dali/array/op2/dot.h"
#include "dali/array/op2/swapaxes.h"
#include "dali/array/op2/reshape.h"
#include "dali/array/op2/spatial/im2col.h"
#include "dali/array/op2/spatial/data_format_helper.h"
#include "dali/array/op/spatial/utils.h"

#include "dali/utils/assert2.h"
#include "dali/utils/make_message.h"


namespace expression {
    struct Conv2DExpressionState: public RValue {
        std::shared_ptr<const RValue> input_;
        std::shared_ptr<const RValue> filters_;
        int stride_h_;
        int stride_w_;
        PADDING_T padding_;
        std::string data_format_;

        Conv2DExpressionState(std::shared_ptr<const RValue> input,
                              std::shared_ptr<const RValue> filters,
                              int stride_h,
                              int stride_w,
                              PADDING_T padding,
                              const std::string& data_format) :
                input_(input),
                filters_(filters),
                stride_h_(stride_h),
                stride_w_(stride_w),
                padding_(padding),
                data_format_(data_format) {
        }

        virtual std::string name() const {
            return "conv2d";
        }

        DType dtype() const {
            return input_->dtype();
        }

        std::vector<int> bshape() const {
            auto input_shape = input_->shape();
            if (input_shape.size() == 3) {
                // make the shape appear to have N-dimension be 1
                // if input is 3D:
                input_shape.insert(input_shape.begin() + data_format_.find('N'), 1);
            }
            auto info = internal::compute_conv_info(
                input_shape, filters_->shape(),
                stride_h_, stride_w_, padding_, data_format_
            );
            std::vector<int> out_shape(input_shape.size(), 0);
            int n_dim = data_format_.find('N');
            int h_dim = data_format_.find('H');
            int w_dim = data_format_.find('W');
            int c_dim = data_format_.find('C');
            if (input_shape.size() == 3) {
                // correct dimensions if input is 3D by
                // offsetting accordingly if N shows up
                // before the relevant dimension:
                if (n_dim < w_dim) w_dim = w_dim - 1;
                if (n_dim < h_dim) h_dim = h_dim - 1;
                if (n_dim < c_dim) c_dim = c_dim - 1;
            }
            out_shape[c_dim] = info.out_channels;
            out_shape[h_dim] = info.out_h;
            out_shape[w_dim] = info.out_w;
            // make the output shape show a batch size only if input was 4D:
            if (input_shape.size() == 4) {
                out_shape[n_dim] = info.batch_size;
            }
            return out_shape;
        }

        int ndim() const {
            return input_->ndim();
        }

        std::vector<std::shared_ptr<const ExpressionState>> arguments() const {
            return {input_, filters_};
        }

        std::shared_ptr<const Runnable> assign_to(std::shared_ptr<const LValue> op, memory::Device device) const {
            // currently all convolutions using im2col + matmul
            auto input_shape = input_->shape();
            if (input_shape.size() == 3) {
                // make the shape appear to have N-dimension be 1
                // if input is 3D:
                input_shape.insert(input_shape.begin() + data_format_.find('N'), 1);
            }
            auto info = internal::compute_conv_info(
                input_shape, filters_->shape(),
                stride_h_,
                stride_w_,
                padding_,
                data_format_);
            int n_dim = data_format_.find('N');
            int h_dim = data_format_.find('H');
            int w_dim = data_format_.find('W');
            int c_dim = data_format_.find('C');
            // TODO add these transformations:
            auto filters_nxxx = op::swapaxes(Expression(filters_), n_dim, 0);
            auto filters_nX = op::reshape(filters_nxxx, {filters_nxxx.shape()[0], -1});
            return op::dot2(
                op::im2col(
                    Expression(input_),
                    info.filter_h,
                    info.filter_w,
                    stride_h_,
                    stride_w_,
                    /*padding_h=*/2 * info.padding_h + info.odd_padding_h,
                    /*padding_w=*/2 * info.padding_w + info.odd_padding_w,
                    data_format_
                ),
                filters_nX
            ).state_->as_rvalue()->assign_to(op, device);
        }
    };
}  // namespace expression

namespace op {
    expression::Expression conv2d(const expression::Expression& input,
                                  const expression::Expression& filters,
                                  int stride_h,
                                  int stride_w,
                                  PADDING_T padding,
                                  const std::string& data_format) {
        ASSERT2(input.ndim() == 3 || input.ndim() == 4, utils::make_message(
            "Input argument to conv2d must be 3D or 4D (got input.ndim=",
            input.ndim(), ")."));
        ASSERT2(filters.ndim() == 4, utils::make_message(
            "Filters argument to conv2d must be 4D (got filters.ndim=",
            filters.ndim(), ")."));
        auto input_rvalue  = input.state_->as_rvalue();
        auto filters_rvalue = filters.state_->as_rvalue();
        ASSERT2(input_rvalue, "Input (1st argument) to conv2d must be a rvalue.");
        ASSERT2(filters_rvalue, "Filters (2nd argument) to conv2d must be a rvalue.");
        int n_dim, c_dim, h_dim, w_dim;
        check_data_format(data_format, &n_dim, &c_dim, &h_dim, &w_dim);
        // TODO(szymon): add type promotion.
        // TODO: check that filters sizes lines up with image size
        return expression::Expression(
            std::make_shared<expression::Conv2DExpressionState>(
                input_rvalue,
                filters_rvalue,
                stride_h,
                stride_w,
                padding,
                data_format
            )
        );
    }
}  // namespace op