#include "col2im.h"
#include <map>
#include "dali/array/op/spatial_utils.h"
#include "dali/array/jit/jit.h"
#include "dali/array/jit/jit_utils.h"
#include "dali/array/op/elementwise_operation.h"
#include "dali/utils/make_message.h"

namespace {
    inline int get_channel_dim(const std::vector<int>& image_shape,
                              const std::string& data_format, char c) {
        if (image_shape.size() == 3) {
            // no n dimension
            return data_format.find(c) - 1;
        } else {
            // 4 dimensions map to data_format adequately
            return data_format.find(c);
        }
    }

    inline int get_image_dim(const std::vector<int>& image_shape,
                             const std::string& data_format, char c) {
        return image_shape[get_channel_dim(image_shape, data_format, c)];
    }
}

namespace op {
    namespace jit {
        struct Col2Im : public JITNode {
            std::string data_format_;
            Col2Im(const Array& input,
                   const Array& filter_h,
                   const Array& filter_w,
                   const Array& stride_h,
                   const Array& stride_w,
                   const Array& dilate_h,
                   const Array& dilate_w,
                   const Array& prepad_h,
                   const Array& prepad_w,
                   const Array& postpad_h,
                   const Array& postpad_w,
                   const std::vector<int>& image_shape,
                   const std::string& data_format) :
                    JITNode(image_shape, input.dtype(),
                        {input, filter_h, filter_w, stride_h, stride_w, dilate_h,
                         dilate_w, prepad_h, prepad_w, postpad_h, postpad_w}),
                    data_format_(data_format) {}

            expression_ptr copy() const override {
                return std::make_shared<Col2Im>(
                    arguments_[0], arguments_[1],
                    arguments_[2], arguments_[3],
                    arguments_[4], arguments_[5],
                    arguments_[6], arguments_[7],
                    arguments_[8], arguments_[9],
                    arguments_[10], shape_, data_format_);
            }

            void prefix_code(memory::DeviceT device_type, insert_t insert) const override {
                int computation_rank = ndim();
                int c_dim = data_format_.find('C'),
                    w_dim = data_format_.find('W'),
                    h_dim = data_format_.find('H'),
                    n_dim = data_format_.find('N');

                c_dim = computation_rank + c_dim - 4;
                w_dim = computation_rank + w_dim - 4;
                h_dim = computation_rank + h_dim - 4;
                n_dim = computation_rank + n_dim - 4;

                std::map<unsigned char, std::string> letter2symbol = {
                    {'C', "c"},
                    {'W', "((w - pw * stride_w_[0]) / dilate_w_[0])"},
                    {'H', "((h - ph * stride_h_[0]) / dilate_h_[0])"}
                };
                std::map<unsigned char, std::string> letter2stride = {
                    {'C', "o_channel_"},
                    {'W', "filter_w_[0]"},
                    {'H', "filter_h_[0]"}
                };
                auto stride_1_letter = n_dim == 3 ? data_format_[2] : data_format_[3];
                auto stride_2_letter = n_dim == 2 ? data_format_[1] : data_format_[2];
                auto stride_3_letter = n_dim == 1 ? data_format_[0] : data_format_[1];

                auto access_im2coled_data = utils::make_message(
                    // stride of lower 2 dimensions
                    letter2symbol.at(stride_3_letter), " * (", letter2stride.at(stride_2_letter), " * ", letter2stride.at(stride_1_letter), ") + ",
                    // stride of lower 1 dimension
                    letter2symbol.at(stride_2_letter), " * ", letter2stride.at(stride_1_letter), " + ",
                    // stride is 1:
                    letter2symbol.at(stride_1_letter)
                );

                // TODO(jonathan): ideally these variable would be computed outside the hot loop once
                std::string o_height_width_channel_ = utils::make_message(
                    "const int o_height_ = (shape_[", get_channel_dim(shape_, data_format_, 'H'), "] + prepad_h_[0] + postpad_h_[0] - (dilate_h_[0] * (filter_h_[0] - 1) + 1)) / stride_h_[0] + 1;\n"
                    "const int o_width_ = (shape_[", get_channel_dim(shape_, data_format_, 'W'), "] + prepad_w_[0] + postpad_w_[0] - (dilate_w_[0] * (filter_w_[0] - 1) + 1)) / stride_w_[0] + 1;\n"
                    "const int& o_channel_ = shape_[", get_channel_dim(shape_, data_format_, 'C'), "];\n"
                );

                std::string kernel = utils::make_message(
                    "const int c = query[", c_dim, "];\n"
                    "int w = query[", w_dim, "];\n"
                    "int h = query[", h_dim, "];\n"
                    "const int n = ", computation_rank == 3 ? "1" : utils::make_message("query[", n_dim, "]"), ";\n",
                    o_height_width_channel_,
                    "\n"
                    "h += prepad_h_[0];\n"
                    "w += prepad_w_[0];\n"
                    "\n"
                    "const int patch_size_h_dilate = (dilate_h_[0] * (filter_h_[0] - 1) + 1);\n"
                    "const int patch_size_w_dilate = (dilate_w_[0] * (filter_w_[0] - 1) + 1);\n"
                    "\n"
                    "const int ph_min =\n"
                    "    h < patch_size_h_dilate ? h % dilate_h_[0] : (h - patch_size_h_dilate + stride_h_[0]) / stride_h_[0];\n"
                    "\n"
                    "const int pw_min =\n"
                    "    w < patch_size_w_dilate ? w % dilate_w_[0] : (w - patch_size_w_dilate + stride_w_[0]) / stride_w_[0];\n"
                    "const int ph_max = int_min((h + stride_h_[0]) / stride_h_[0], o_height_);\n"
                    "const int pw_max = int_min((w + stride_w_[0]) / stride_w_[0], o_width_);\n"
                    "T res = static_cast<T>(0);\n"
                    "for (int ph = ph_min; ph < ph_max; ph += dilate_h_[0]) {\n"
                    "    for (int pw = pw_min; pw < pw_max; pw += dilate_w_[0]) {\n"
                    "        res += input_[{\n",
                                access_im2coled_data, ", (n * o_height_ + ph) * o_width_ + pw\n"
                    "        }];\n"
                    "    }\n"
                    "}\n"
                    "return res;\n");

                define_kernel(/*ndim=*/ndim(),
                              /*has_shape=*/true,
                              /*arguments=*/{"input",
                                             "filter_h", "filter_w",
                                             "stride_h", "stride_w",
                                             "dilate_h", "dilate_w",
                                             "prepad_h", "prepad_w",
                                             "postpad_h", "postpad_w"},
                              /*kernel=*/kernel,
                              /*name=*/kernel_name(),
                              /*is_assignable=*/false,
                              insert);
            }

            virtual void compilation_parameters(utils::Hasher& hasher) const override {
                hasher.add(data_format_);
            }

            virtual bool shape_required() const override {return true;}

            std::string kernel_name() const {
                return utils::make_message("col2im_", data_format_, "_", ndim(), "d");
            }

            std::string get_call_code_nd(const SymbolTable& symbol_table,
                                         memory::DeviceT device_type) const override {
                return generate_call_code_nd(this, kernel_name(),
                                             symbol_table, device_type,
                                             /*has_shape=*/true);
            }
        };
    }  // namespace jit

    Array col2im(const Array& input,
                 const std::vector<int>& image_shape,
                 int filter_h,
                 int filter_w,
                 int stride_h,
                 int stride_w,
                 const std::string& data_format) {
        ASSERT2(input.ndim() == 2, utils::make_message("col2im takes a 2D "
            "input (got ", input.ndim(), ")."));
        int image_ndim = image_shape.size();
        ASSERT2(image_ndim == 3 || image_ndim == 4, utils::make_message(
            "col2im's image_shape should have size == 3 or 4 (got ", image_shape, ")."));
        int n_dim, c_dim, h_dim, w_dim;
        check_data_format(data_format, &n_dim, &c_dim, &h_dim, &w_dim);
        // No N dimension if image is 3D:
        if (image_ndim == 3) {
            if (n_dim < w_dim) w_dim = w_dim - 1;
            if (n_dim < h_dim) h_dim = h_dim - 1;
            if (n_dim < c_dim) c_dim = c_dim - 1;
        }
        const int image_w = image_shape[w_dim];
        const int image_h = image_shape[h_dim];
        bool image_w_gt_patch = image_w >= filter_w;
        bool image_h_gt_patch = image_h >= filter_h;
        ASSERT2(image_w_gt_patch && image_h_gt_patch, utils::make_message("the "
            "image shape of col2im with data_format=", data_format, " should be "
            "smaller than filter size (filter_h=", filter_h, " vs. image_h=",
            image_h, ", filter_w=", filter_w, " vs. w_dim=", image_w, ")."));

        const int& i_channel = image_shape[c_dim];
        const int& i_height  = image_shape[h_dim];
        const int& i_width   = image_shape[w_dim];

        std::vector<int> input_shape = input.shape();

        ASSERT2(filter_h * filter_w * i_channel == input_shape[0],
            utils::make_message("col2im with data_format ", data_format, "'s "
            "input must have its first dimension be filter_h * filter_w * "
            "i_channel = ", filter_h * filter_w * i_channel, " = filter_h [",
            filter_h, "] * filter_w [", filter_w, "] * i_channel [",
            i_channel, "] (got ", input_shape[0], ")."));
        // calculate batch size
        const int batch_size = image_ndim == 3 ? 1 : image_shape[n_dim];
        const int o_height = (image_shape[h_dim] - ((filter_h - 1) + 1)) / stride_h + 1;
        const int o_width  = (image_shape[w_dim] - ((filter_w - 1) + 1)) / stride_w + 1;

        ASSERT2(o_height * o_width * batch_size == input_shape[1],
            utils::make_message("col2im with data_format ", data_format, "'s "
            " input must have its second dimension be ",
            o_height * o_width * batch_size, " = {(i_height - filter_h) / stride_h_ "
            "+ 1} [", o_height, "] * {(i_width  - filter_w) / stride_w + 1} [",
            o_width, "] * batch_size [", batch_size, "] (got ", input_shape[1], ")."));

        return Array(
            std::make_shared<jit::Col2Im>(
                input,
                filter_h,
                filter_w,
                stride_h,
                stride_w,
                /*dilate_h=*/1,
                /*dilate_w=*/1,
                /*prepad_h=*/0,
                /*prepad_w=*/0,
                /*postpad_h=*/0,
                /*postpad_w=*/0,
                image_shape,
                data_format
            )
        );
    }
}  // namespace op
