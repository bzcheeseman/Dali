#include "im2col.h"
#include <map>
#include "dali/array/op/spatial_utils.h"
#include "dali/array/jit/jit.h"
#include "dali/array/jit/jit_utils.h"
#include "dali/array/op/elementwise_operation.h"
#include "dali/utils/make_message.h"

namespace op {
    namespace jit {
        struct Im2Col : public JITNode {
            std::string data_format_;

            Im2Col(const Array& image,
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
                   const std::vector<int>& im2col_shape,
                   const std::string& data_format) :
                    JITNode(im2col_shape, image.dtype(),
                        {image,
                         filter_h,
                         filter_w,
                         stride_h,
                         stride_w,
                         dilate_h,
                         dilate_w,
                         prepad_h,
                         prepad_w,
                         postpad_h,
                         postpad_w}), data_format_(data_format) {
            }

            expression_ptr copy() const override {
                return std::make_shared<Im2Col>(
                    arguments_[0],
                    arguments_[1],
                    arguments_[2],
                    arguments_[3],
                    arguments_[4],
                    arguments_[5],
                    arguments_[6],
                    arguments_[7],
                    arguments_[8],
                    arguments_[9],
                    arguments_[10],
                    shape_,
                    data_format_
                );
            }

            void prefix_code(memory::DeviceT device_type, insert_t insert) const override {
                int c_dim = data_format_.find('C'),
                    w_dim = data_format_.find('W'),
                    h_dim = data_format_.find('H'),
                    n_dim = data_format_.find('N');
                std::string access_image = utils::make_message(
                    "[{",
                    (unsigned char)std::tolower((unsigned char)data_format_[0]),
                    ", ",
                    (unsigned char)std::tolower((unsigned char)data_format_[1]),
                    ", ",
                    (unsigned char)std::tolower((unsigned char)data_format_[2]),
                    ", ",
                    (unsigned char)std::tolower((unsigned char)data_format_[3]),
                    "}]"
                );

                // obtain the values c, h, and w by iteratively chipping away at the
                // index using strides:
                auto stride_1_letter = n_dim == 3 ? data_format_[2] : data_format_[3];
                auto stride_2_letter = n_dim == 2 ? data_format_[1] : data_format_[2];
                auto stride_3_letter = n_dim == 1 ? data_format_[0] : data_format_[1];

                std::map<unsigned char, std::string> letter2name = {
                    {'C', "c"}, {'W', "w_offset"}, {'H', "h_offset"}
                };
                std::map<unsigned char, std::string> letter2stride = {
                    {'C', utils::make_message("image_.shape()[", c_dim, "]")},
                    {'W', "filter_w_[0]"}, {'H', "filter_h_[0]"}
                };
                std::map<unsigned char, std::string> letter2adj = {
                    {'C', ""}, {'H', " * dilate_h_[0]"}, {'W', " * dilate_w_[0]"}
                };

                auto compute_whc = utils::make_message(
                    // obtain offset in the relevant dimension:
                    "const int ", letter2name.at(stride_1_letter), " = query[0] % ",
                    letter2stride.at(stride_1_letter), letter2adj.at(stride_1_letter), ";\n"
                    // obtain index post-this offset computation:
                    "const int index_without_", letter2name.at(stride_1_letter), " = query[0] / ",
                    letter2stride.at(stride_1_letter), ";\n"
                    // obtain index for next dimension:
                    "const int ", letter2name.at(stride_2_letter), " = index_without_",
                    letter2name.at(stride_1_letter), " % ", letter2stride.at(stride_2_letter), letter2adj.at(stride_2_letter), ";\n"
                    // obtain index post-this offset computation:
                    "const int index_without_", letter2name.at(stride_2_letter), " = index_without_",
                    letter2name.at(stride_1_letter), " / ", letter2stride.at(stride_2_letter), ";\n"
                    // obtain index for last dimension:
                    "const int ", letter2name.at(stride_3_letter), " = index_without_",
                    letter2name.at(stride_2_letter), " % ", letter2stride.at(stride_3_letter), letter2adj.at(stride_3_letter), ";\n"
                    // now use the collected values to compute w and h:
                    "const int w = (query[1] % o_width_) * stride_w_[0] + w_offset - prepad_w_[0];\n"
                    "const int jdivw = query[1] / o_width_;\n"
                    "const int h = (jdivw % o_height_) * stride_h_[0] + h_offset - prepad_h_[0];\n"
                );

                std::string kernel = utils::make_message(
                    "const int o_height_ = (image_.shape()[", h_dim, "] + prepad_h_[0] + postpad_h_[0] - (dilate_h_[0] * (filter_h_[0] - 1) + 1)) / stride_h_[0] + 1;\n"
                    "const int o_width_ = (image_.shape()[", w_dim, "] + prepad_w_[0] + postpad_w_[0] - (dilate_w_[0] * (filter_w_[0] - 1) + 1)) / stride_w_[0] + 1;\n"
                    "const int n = query[1] / (o_height_ * o_width_);\n",
                     compute_whc,
                    "if (0 <= w && w < image_.shape()[", w_dim, "] && 0 <= h && h < image_.shape()[", h_dim, "]) {\n"
                    "    return image_", access_image, ";\n"
                    "} else {\n"
                    "//    padding with zeros:\n"
                    "    return T(0.0f);\n"
                    "}\n"
                );

                define_kernel(/*ndim=*/ndim(),
                              /*has_shape=*/true,
                              /*arguments=*/{"image",
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

            std::string kernel_name() const {
                return utils::make_message("im2col_", data_format_, "_", ndim(), "d");
            }

            virtual void compilation_parameters(utils::Hasher& hasher) const override {
                hasher.add(data_format_);
            }

            virtual bool shape_required() const override {return true;}

            std::string get_call_code_nd(const SymbolTable& symbol_table,
                                         memory::DeviceT device_type) const override {
                return generate_call_code_nd(this,
                                             kernel_name(),
                                             symbol_table, device_type,
                                             true);
            }
        };
    }  // namespace jit

    std::vector<int> im2col_shape(
            const std::vector<int>& src_shape,
            const int& filter_h,
            const int& filter_w,
            const int& stride_h,
            const int& stride_w,
            const int& dilate_h,
            const int& dilate_w,
            const int& prepad_h,
            const int& prepad_w,
            const int& postpad_h,
            const int& postpad_w,
            const std::string& data_format) {

        int w_dim = data_format.find('W'),
            h_dim = data_format.find('H'),
            c_dim = data_format.find('C'),
            n_dim = data_format.find('N');

        // No N dimension if image is 3D:
        if (src_shape.size() == 3) {
            w_dim = w_dim - 1;
            h_dim = h_dim - 1;
            c_dim = c_dim - 1;
        }

        const int i_channel = src_shape[c_dim];
        const int i_height  = src_shape[h_dim] + prepad_h + postpad_h;
        const int i_width   = src_shape[w_dim] + prepad_w + postpad_w;
        // calculate number of batches
        const int num = src_shape.size() == 4 ? src_shape[n_dim] : 1;
        const int o_height = (i_height - (dilate_h * (filter_h - 1) + 1)) / stride_h + 1;
        const int o_width  = (i_width  - (dilate_w * (filter_w - 1) + 1)) / stride_w + 1;

        return {
            filter_h * filter_w * i_channel,
            o_height * o_width * num
        };
    }

    Array im2col(const Array& image,
                 int filter_h,
                 int filter_w,
                 int stride_h,
                 int stride_w,
                 int padding_h,
                 int padding_w,
                 int postpad_h,
                 int postpad_w,
                 const std::string& data_format) {
        int image_ndim = image.ndim();
        ASSERT2(image_ndim == 3 || image_ndim == 4, utils::make_message(
            "im2col takes an image with ndim == 3 or ndim == 4 (got ndim=",
            image_ndim, ")."));
        int n_dim, c_dim, h_dim, w_dim;
        check_data_format(data_format, &n_dim, &c_dim, &h_dim, &w_dim);
        // No N dimension if image is 3D:
        if (image_ndim == 3) {
            if (n_dim < w_dim) w_dim = w_dim - 1;
            if (n_dim < h_dim) h_dim = h_dim - 1;
        }
        auto image_bshape = image.shape();
        const int image_w = image_bshape[w_dim];
        const int image_h = image_bshape[h_dim];
        bool image_w_gt_patch = image_w >= filter_w;
        bool image_h_gt_patch = image_h >= filter_h;
        ASSERT2(image_w_gt_patch && image_h_gt_patch, utils::make_message("the "
            "image shape of im2col with data_format=", data_format, " should be "
            "smaller than filter size (filter_h=", filter_h, " vs. image_h=",
            image_h, ", filter_w=", filter_w, " vs. w_dim=", image_w, ")."));
        ASSERT2(padding_h >= 0, utils::make_message(
            "padding_h should be a positive value (got ", padding_h, ")."));
        ASSERT2(padding_w >= 0, utils::make_message(
            "padding_w should be a positive value (got ", padding_w, ")."));
        ASSERT2(postpad_h >= 0, utils::make_message(
            "postpad_h should be a positive value (got ", postpad_h, ")."));
        ASSERT2(postpad_w >= 0, utils::make_message(
            "postpad_w should be a positive value (got ", postpad_w, ")."));

        int dilate_h = 1, dilate_w = 1;

        return Array(std::make_shared<jit::Im2Col>(
            image,
            filter_h,
            filter_w,
            stride_h,
            stride_w,
            dilate_h,
            dilate_w,
            padding_h,
            padding_w,
            postpad_h,
            postpad_w,
            im2col_shape(image.shape(),
                         filter_h,
                         filter_w,
                         stride_h,
                         stride_w,
                         dilate_h,
                         dilate_w,
                         padding_h,
                         padding_w,
                         postpad_h,
                         postpad_w,
                         data_format),
            data_format));
    }
}  // namespace op
