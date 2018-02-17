#include "conv.h"
#include "dali/array/op/im2col.h"
#include "dali/array/op/col2im.h"
#include "dali/array/op/dot.h"
#include "dali/array/op/spatial_utils.h"
#include "dali/array/jit/jit_runner.h"
#include "dali/utils/make_message.h"
#include "dali/utils/assert2.h"

namespace op {
    namespace jit {
        struct Pool2d : public JITNode {
            const PADDING_T padding_;
            const std::string functor_name_;
            const bool nchw_;

            // MUST IMPLEMENT
            std::string kernel_name() const {
                return utils::make_message("pool2d_", nchw_ ? "nchw" : "nhwc");
            }

            // DO NOT REIMPLEMENT
            Pool2d(const Array& argument,
                   const Array& window_h, const Array& window_w,
                   const Array& stride_h, const Array& stride_w,
                   PADDING_T padding,
                   const std::string& functor_name, bool nchw, const std::vector<int>& shape) :
                JITNode(shape, argument.dtype(), {argument, window_h, window_w, stride_h, stride_w}),
                padding_(padding), functor_name_(functor_name), nchw_(nchw) {
            }

            expression_ptr copy() const override {
                return std::make_shared<Pool2d>(arguments_[0], arguments_[1], arguments_[2], arguments_[3], arguments_[4], padding_, functor_name_, nchw_, shape_);
            }

            virtual void compilation_parameters(utils::Hasher& hasher) const override {
                hasher.add(int(padding_)).add(functor_name_).add(nchw_);
            }

            void prefix_code(memory::DeviceT device_type, insert_t insert) const override {
                std::string clsname = utils::make_message("Pool2D", nchw_ ? "NCHW" : "NHWC");
                // TODO(jonathan): generalize to ND-pooling & any ordering of channels/spatial
                insert(utils::make_message(
                    "template<typename Reducer, typename Type, typename C1, typename C2, typename C3, typename C4, typename C5>\n"
                    "struct ", clsname, " {\n"
                    "    C1 arg_;\n"
                    "    C2 window_h_;\n"
                    "    C3 window_w_;\n"
                    "    C4 stride_h_;\n"
                    "    C5 stride_w_;\n"
                    "    static const int ndim = C1::ndim;\n"
                    "    typedef Type T;\n"
                    "    XINLINE Shape<ndim> shape() const {\n"
                    "        return arg_.shape();\n"
                    "    }\n"
                    "    XINLINE ", clsname, "(C1 arg, C2 window_h, C3 window_w, C4 stride_h, C5 stride_w) :\n"
                    "        arg_(arg), window_h_(window_h), window_w_(window_w), stride_h_(stride_h), stride_w_(stride_w) {}\n"
                    "    XINLINE T operator[](const Shape<ndim>& query) const {\n"
                    "        T src = arg_[query];\n"
                    "        return 0;\n"
                    "    }\n"
                    "};\n"));
                insert(utils::make_message(
                    "template<typename Reducer, typename Type, typename C1, typename C2, typename C3, typename C4, typename C5>\n"
                    "XINLINE ", clsname, "<Reducer, Type, C1, C2, C3, C4, C5> ", kernel_name(), "(C1 arg, C2 window_h, C3 window_w, C4 stride_h, C5 stride_w) {\n"
                    "    return ", clsname, "<Reducer, Type, C1, C2, C3, C4, C5>(arg, window_h, window_w, stride_h, stride_w);\n"
                    "}\n"));
            }

            virtual std::string get_call_code_nd(const SymbolTable& symbol_table,
                                                 memory::DeviceT device_type) const override {
                return utils::make_message(
                    kernel_name(), "<", functor_name_, ", " , dtype_to_cpp_name(dtype_), ">(",
                    op::jit::get_call_code_nd(arguments_[0], symbol_table, device_type), ",",
                    op::jit::get_call_code_nd(arguments_[1], symbol_table, device_type), ",",
                    op::jit::get_call_code_nd(arguments_[2], symbol_table, device_type), ",",
                    op::jit::get_call_code_nd(arguments_[3], symbol_table, device_type), ",",
                    op::jit::get_call_code_nd(arguments_[4], symbol_table, device_type), ")");
            }
        };
        struct Pool2dBackward : public JITNode {
            const PADDING_T padding_;
            const std::string functor_name_;
            const bool nchw_;

            Pool2dBackward(const Array& y, const Array& dy, const Array& x,
                           const Array& window_h, const Array& window_w,
                           const Array& stride_h, const Array& stride_w,
                           PADDING_T padding,
                           const std::string& functor_name, bool nchw) :
                JITNode(x.shape(), y.dtype(), {y, dy, x, window_h, window_w, stride_h, stride_w}),
                padding_(padding), functor_name_(functor_name), nchw_(nchw) {
            }

            virtual void compilation_parameters(utils::Hasher& hasher) const override {
                hasher.add(int(padding_)).add(functor_name_).add(nchw_);
            }

            expression_ptr copy() const override {
                return std::make_shared<Pool2dBackward>(arguments_[0], arguments_[1], arguments_[2], arguments_[3], arguments_[4], arguments_[5], arguments_[6], padding_, functor_name_, nchw_);
            }

            std::string kernel_name() const {
                return utils::make_message("pool2d_backward_", nchw_ ? "nchw" : "nhwc");
            }

            void prefix_code(memory::DeviceT device_type, insert_t insert) const override {
                std::string clsname = utils::make_message("Pool2DBackward", nchw_ ? "NCHW" : "NHWC");
                // TODO(jonathan): generalize to ND-pooling & any ordering of channels/spatial
                // "        for (int py = py_min; py < py_max; ++py) {\n"
                //     "             for (int px = px_min; px < px_max; ++px) {\n"
                //     "                 val += Reducer::PartialGrad(vsrc, y_[query_shifted], dy_[query_shifted]);\n"
                //     "             }\n"
                //     "        }\n"
                insert(utils::make_message(
                    "template<typename Reducer, typename Type, typename C1, typename C2, typename C3, typename C4, typename C5, typename C6, typename C7>\n"
                    "struct ", clsname, " {\n"
                    "    C1 y_;\n"
                    "    C2 dy_;\n"
                    "    C3 x_;\n"
                    "    C4 window_h_;\n"
                    "    C5 window_w_;\n"
                    "    C6 stride_h_;\n"
                    "    C7 stride_w_;\n"
                    "    static const int ndim = C3::ndim;\n"
                    "    typedef Type T;\n"
                    "    XINLINE Shape<ndim> shape() const {\n"
                    "        return x_.shape();\n"
                    "    }\n"
                    "    XINLINE ", clsname, "(C1 y, C2 dy, C3 x, C4 window_h, C5 window_w, C6 stride_h, C7 stride_w)\n"
                    "        : y_(y), dy_(dy), x_(x), window_h_(window_h), window_w_(window_w), stride_h_(stride_h), stride_w_(stride_w) {}\n"
                    "    XINLINE T operator[](const Shape<ndim>& query) const {\n"
                    "        T src = x_[query];\n"
                    "        return 0;\n"
                    "    }\n"
                    "};\n"));
                insert(utils::make_message(
                    "template<typename Reducer, typename Type, typename C1, typename C2, typename C3, typename C4, typename C5, typename C6, typename C7>\n"
                    "XINLINE ", clsname, "<Reducer, Type, C1, C2, C3, C4, C5, C6, C7> ", kernel_name(), "(C1 y, C2 dy, C3 x, C4 window_h, C5 window_w, C6 stride_h, C7 stride_w) {\n"
                    "    return ", clsname, "<Reducer, Type, C1, C2, C3, C4, C5, C6, C7>(y, dy, x, window_h, window_w, stride_h, stride_w);\n"
                    "}\n"));
            }

            virtual std::string get_call_code_nd(const SymbolTable& symbol_table,
                                                 memory::DeviceT device_type) const override {
                return utils::make_message(
                    kernel_name(), "<", functor_name_, ", " , dtype_to_cpp_name(dtype_), ">(",
                    op::jit::get_call_code_nd(arguments_[0], symbol_table, device_type), ",",
                    op::jit::get_call_code_nd(arguments_[1], symbol_table, device_type), ",",
                    op::jit::get_call_code_nd(arguments_[2], symbol_table, device_type), ",",
                    op::jit::get_call_code_nd(arguments_[3], symbol_table, device_type), ",",
                    op::jit::get_call_code_nd(arguments_[4], symbol_table, device_type), ",",
                    op::jit::get_call_code_nd(arguments_[5], symbol_table, device_type), ",",
                    op::jit::get_call_code_nd(arguments_[6], symbol_table, device_type), ")");
            }
        };
    }

    #define DALI_CHECK_DATA_FORMAT(name, data_format)\
        ASSERT2(data_format == "NCHW" | data_format == "NHWC", utils::make_message(\
            name "'s data_format must be NCHW or NHWC but got ", data_format,\
            " instead."));
    #define DALI_CHECK_NDIM(name, array_name, array)\
        ASSERT2(( array ).ndim() == 4, utils::make_message(\
            name "'s " array_name " must be 4 dimensional but got " array_name " ",\
            ( array ).full_expression_name(), " with ndim = ", ( array ).ndim(), "."));

    Array conv2d(const Array& input,
                 const Array& filters,
                 int stride_h,
                 int stride_w,
                 PADDING_T padding,
                 const std::string& data_format) {
        ASSERT2(input.ndim() == 3 || input.ndim() == 4, utils::make_message(
            "Input argument to conv2d must be 3D or 4D (got input.ndim=",
            input.ndim(), ")."));
        DALI_CHECK_NDIM("conv2d", "filters", filters);
        int n_dim, c_dim, h_dim, w_dim;
        check_data_format(data_format, &n_dim, &c_dim, &h_dim, &w_dim);

        auto input_shape = input.shape();
        if (input_shape.size() == 3) {
            // make the shape appear to have N-dimension be 1
            // if input is 3D:
            input_shape.insert(input_shape.begin() + n_dim, 1);
        }
        auto info = compute_conv2d_info(
            input_shape, filters.shape(),
            stride_h,
            stride_w,
            padding,
            data_format);

        auto filters_nxxx = filters.swapaxes(n_dim, 0);
        auto filters_nX = filters_nxxx.reshape({filters_nxxx.shape()[0], -1});

        // format is c * filt_h * filt_w x batch x out_h x out_w
        auto im2col_image = op::im2col(
            input,
            info.filter_h,
            info.filter_w,
            stride_h,
            stride_w,
            /*padding_h=*/info.padding_h,
            /*padding_w=*/info.padding_w,
            /*postpad_h=*/info.padding_h + info.odd_padding_h,
            /*postpad_w=*/info.padding_w + info.odd_padding_w,
            data_format);
        if (data_format == "NCHW") {
            return op::dot(filters_nX, im2col_image).reshape(
                {info.out_channels, info.batch_size, info.out_h, info.out_w}).swapaxes(0, 1);
        } else if (data_format == "NHWC") {
            return op::dot(im2col_image.transpose(),
                           filters_nX.transpose()).reshape(
                {info.batch_size, info.out_h, info.out_w, info.out_channels});
        } else {
            ASSERT2(false, utils::make_message(
                "conv2d only supports data_format NCHW and NHWC "
                "(got data_format = ", data_format, ")."));
        }
    }

    Array conv2d_backward_input(Array filters,
                                Array out_dw,
                                int stride_h,
                                int stride_w,
                                const std::vector<int>& result_shape,
                                PADDING_T padding,
                                const std::string& data_format) {
        DALI_CHECK_NDIM("conv2d_backward_input", "out_dw", out_dw);
        DALI_CHECK_NDIM("conv2d_backward_input", "filters", filters);
        DALI_CHECK_DATA_FORMAT("conv2d_backward_input", data_format);
        ASSERT2(result_shape.size() == 4, utils::make_message(
            "conv2d_backward_input's result_shape must be of size 4, "
            "but got ", result_shape, "."));
       auto info = op::compute_conv2d_info(result_shape,
                                           filters.shape(),
                                           stride_h,
                                           stride_w,
                                           padding,
                                           data_format);
        filters = filters.reshape({filters.shape()[0], -1}).transpose();
        if (data_format == "NCHW") {
            auto out_dw_cnhw = out_dw.swapaxes(0, 1);
            out_dw_cnhw = out_dw_cnhw.reshape({out_dw_cnhw.shape()[0], -1});
            // filters2D?
            return op::col2im(op::dot(filters, out_dw_cnhw),
                              result_shape,
                              info.filter_h,
                              info.filter_w,
                              stride_h,
                              stride_w,
                              data_format);
        } else {
            /* NHWC forward pass is:
             *
             *   output = (Im2col(Input))^T * Filters^T
             *
             * NHWC backward pass is:
             *
             *   ∂Im2col(Input)/∂E = Filters^T * ∂output/∂E^T
             *
             * Our 2d shapes into gemm are as follows:
             *
             *   ∂Im2col(Input)/∂E => (window_h * window_w * c) x (n * h * w)
             *
             *   Filters^T => (window_h * window_w * c) x (channels_out)
             *
             *   ∂output/∂E^T => (channels_out) x (n * h * w)
             *
             */
            out_dw = out_dw.reshape({-1, out_dw.shape()[3]}).transpose();
            return op::col2im(op::dot(filters, out_dw),
                              result_shape,
                              info.filter_h,
                              info.filter_w,
                              stride_h,
                              stride_w,
                              data_format);
        }
    }

    Array conv2d_backward_filters(Array input,
                                  Array out_dw,
                                  int stride_h,
                                  int stride_w,
                                  const std::vector<int>& filters_shape,
                                  PADDING_T padding,
                                  const std::string& data_format) {
        DALI_CHECK_NDIM("conv2d_backward_filters", "out_dw", out_dw);
        DALI_CHECK_NDIM("conv2d_backward_filters", "input", input);
        DALI_CHECK_DATA_FORMAT("conv2d_backward_filters", data_format);
        ASSERT2(filters_shape.size() == 4, utils::make_message(
            "conv2d_backward_filters's filters_shape must be of size 4, "
            "but got ", filters_shape, "."));
        auto info = op::compute_conv2d_info(input.shape(),
                                            filters_shape,
                                            stride_h,
                                            stride_w,
                                            padding,
                                            data_format);
        auto im2col_image = op::im2col(
            input,
            info.filter_h,
            info.filter_w,
            stride_h,
            stride_w,
            /*padding_h=*/info.padding_h,
            /*padding_w=*/info.padding_w,
            /*postpad_h=*/info.padding_h + info.odd_padding_h,
            /*postpad_w=*/info.padding_w + info.odd_padding_w,
            data_format).transpose();
        if (data_format == "NCHW") {
            auto out_dw_cnhw = out_dw.swapaxes(1, 0);
            out_dw_cnhw = out_dw_cnhw.reshape({out_dw_cnhw.shape()[0], -1});
            // TODO(jonathan) ensure the reshape is done after the computation:
            return op::dot(out_dw_cnhw, im2col_image).reshape(filters_shape);
        } else {
            /* NHWC forward pass is:
             *
             *   output = (Im2col(Input))^T * Filters^T
             *
             * NHWC backward pass is:
             *
             *   ∂Filters/∂E = ∂output/∂E^T * Im2col(Input)^T
             *
             * Our 2d shapes into gemm are as follows:
             *
             *   Im2col(Input)^T => (n * h * w) x (window_h * window_w * c)
             *
             *   ∂output/∂E^T => (channels_out) x (n * h * w)
             *
             *   ∂Filters/∂E =>  (channels_out) x (window_h * window_w * c)
             */
            out_dw = out_dw.reshape({-1, out_dw.shape()[3]}).transpose();
            return op::dot(out_dw, im2col_image).reshape(filters_shape);
        }
    }

    Array pool2d(const Array& input,
                       int window_h,
                       int window_w,
                       int stride_h,
                       int stride_w,
                       POOLING_T pooling_mode,
                       PADDING_T padding,
                       const std::string& data_format) {
        // TODO(jonathan): check window, stride
        DALI_CHECK_NDIM("pool2d", "input", input);
        DALI_CHECK_DATA_FORMAT("pool2d", data_format);
        auto info = op::compute_pool_info(input.shape(),
                                          window_h,
                                          window_w,
                                          stride_h,
                                          stride_w,
                                          padding,
                                          data_format);
        bool nchw = data_format == "NCHW";
        return Array(std::make_shared<op::jit::Pool2d>(
            input, info.window_h, info.window_w, info.stride_h, info.stride_w,
            padding, pooling_mode == POOLING_T_MAX ? "reducers::maximum" : "reducers::avg", nchw,
            nchw ? std::vector<int>({info.batch_size, info.in_channels, info.out_h, info.out_w}) :
            std::vector<int>({info.batch_size, info.out_h, info.out_w, info.in_channels})));
    }

    Array pool2d_backward(const Array& y,
                          const Array& dy,
                          const Array& x,
                          int window_h,
                          int window_w,
                          int stride_h,
                          int stride_w,
                          POOLING_T pooling_mode,
                          PADDING_T padding,
                          const std::string& data_format) {
        DALI_CHECK_NDIM("pool2d_backward", "y", y);
        DALI_CHECK_NDIM("pool2d_backward", "dy", dy);
        DALI_CHECK_NDIM("pool2d_backward", "x", x);
        DALI_CHECK_DATA_FORMAT("cudnn_pool2d_backward", data_format);
        auto info = op::compute_pool_info(x.shape(),
                                          window_h,
                                          window_w,
                                          stride_h,
                                          stride_w,
                                          padding,
                                          data_format);
        bool nchw = data_format == "NCHW";
        return Array(std::make_shared<op::jit::Pool2dBackward>(
            y, dy, x, info.window_h, info.window_w, info.stride_h, info.stride_w,
            padding, pooling_mode == POOLING_T_MAX ? "reducers::maximum" : "reducers::avg", nchw));
    }
}  // namespace op
