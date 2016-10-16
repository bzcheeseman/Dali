#include "im2col.h"

#include "dali/array/op2/operation.h"
#include "dali/array/op2/elementwise_operation.h"
#include "dali/array/op2/rtc_utils.h"
#include "dali/array/op/spatial/utils.h"
#include "dali/utils/hash_utils.h"
#include "dali/utils/make_message.h"

std::vector<int> deduce_im2col_shape(
        const std::vector<int>& src_bshape,
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
        channel_dim = data_format.find('C'),
        batch_dim = data_format.find('N');

    // No N dimension if image is 3D:
    if (src_bshape.size() == 3) {
        w_dim = w_dim - 1;
        h_dim = h_dim - 1;
        channel_dim = channel_dim - 1;
    }

    const int i_channel = src_bshape[channel_dim];
    const int i_height  = src_bshape[h_dim] + prepad_h + postpad_h;
    const int i_width   = src_bshape[w_dim] + prepad_w + postpad_w;
    // calculate number of batches
    const int num = src_bshape.size() == 4 ? src_bshape[batch_dim] : 1;
    const int o_height = (i_height - (dilate_h * (filter_h - 1) + 1)) / stride_h + 1;
    const int o_width  = (i_width  - (dilate_w * (filter_w - 1) + 1)) / stride_w + 1;

    return {
        filter_h * filter_w * i_channel,
        o_height * o_width * num
    };
}


struct Im2ColOperationState : OperationState {
    static const hash_t optype_hash;
    operation_state_ptr image_;

    int filter_h_;
    int filter_w_;

    int stride_h_;
    int stride_w_;

    int dilate_h_;
    int dilate_w_;

    int prepad_h_;
    int prepad_w_;

    int postpad_h_;
    int postpad_w_;

    std::string data_format_;

    operation_state_ptr filter_h_op_;
    operation_state_ptr filter_w_op_;

    operation_state_ptr stride_h_op_;
    operation_state_ptr stride_w_op_;

    operation_state_ptr dilate_h_op_;
    operation_state_ptr dilate_w_op_;

    operation_state_ptr prepad_h_op_;
    operation_state_ptr prepad_w_op_;

    operation_state_ptr postpad_h_op_;
    operation_state_ptr postpad_w_op_;

    Im2ColOperationState(operation_state_ptr image,
                         int filter_h,
                         int filter_w,
                         int stride_h,
                         int stride_w,
                         int dilate_h,
                         int dilate_w,
                         int prepad_h,
                         int prepad_w,
                         int postpad_h,
                         int postpad_w,
                         const std::string& data_format) :
            OperationState(2),
            image_(image),
            filter_h_(filter_h),
            filter_w_(filter_w),
            stride_h_(stride_h),
            stride_w_(stride_w),
            dilate_h_(dilate_h),
            dilate_w_(dilate_w),
            prepad_h_(prepad_h),
            prepad_w_(prepad_w),
            postpad_h_(postpad_h),
            postpad_w_(postpad_w),
            data_format_(data_format),
            // create scalar ops to send constants to kernel:
            filter_h_op_(Operation(filter_h).state_),
            filter_w_op_(Operation(filter_w).state_),
            stride_h_op_(Operation(stride_h).state_),
            stride_w_op_(Operation(stride_w).state_),
            dilate_h_op_(Operation(dilate_h).state_),
            dilate_w_op_(Operation(dilate_w).state_),
            prepad_h_op_(Operation(prepad_h).state_),
            prepad_w_op_(Operation(prepad_w).state_),
            postpad_h_op_(Operation(postpad_h).state_),
            postpad_w_op_(Operation(postpad_w).state_) {
    }

    virtual DType dtype() const {
        return image_->dtype();
    }

    virtual std::string name() const {
        return "im2col";
    }

    virtual std::vector<operation_state_ptr> arguments() const {
        return {
            image_,
            filter_h_op_,
            filter_w_op_,
            stride_h_op_,
            stride_w_op_,
            dilate_h_op_,
            dilate_w_op_,
            prepad_h_op_,
            prepad_w_op_,
            postpad_h_op_,
            postpad_w_op_
        };
    }

    std::string prefix_code(const node_to_info_t& node_to_info) const {
        std::string template_string =
            "template<typename C1, typename C2, "
            "typename C3, typename C4, typename C5, "
            "typename C6, typename C7, typename C8, "
            "typename C9, typename C10, typename C11>\n";
        std::string template_usage =
            "<C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11>";
        int channel_dim = data_format_.find('C'),
            w_dim = data_format_.find('W'),
            h_dim = data_format_.find('H'),
            batch_dim = data_format_.find('N');
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
        return utils::make_message(
            template_string,
            "struct Im2ColKernel", data_format_, " {\n"
            "    const C1 image_;\n"
            "    const int filter_h_;\n"
            "    const int filter_w_;\n"
            "    const int stride_h_;\n"
            "    const int stride_w_;\n"
            "    const int dilate_h_;\n"
            "    const int dilate_w_;\n"
            "    const int prepad_h_;\n"
            "    const int prepad_w_;\n"
            "    const int postpad_h_;\n"
            "    const int postpad_w_;\n"
            "    Shape<C1::ndim> i_shape_;\n"
            "    // derived fields:\n"
            "    const int o_height_;\n"
            "    const int o_width_;\n"
            "    Shape<2> shape_;\n"
            "    static const int ndim = 2;\n"
            "    typedef typename C1::T T;\n"
            "    XINLINE Im2ColKernel", data_format_, "(\n"
            "                         const C1& image,\n"
            "                         const C2& filter_h,\n"
            "                         const C3& filter_w,\n"
            "                         const C4& stride_h,\n"
            "                         const C5& stride_w,\n"
            "                         const C6& dilate_h,\n"
            "                         const C7& dilate_w,\n"
            "                         const C8& prepad_h,\n"
            "                         const C9& prepad_w,\n"
            "                         const C10& postpad_h,\n"
            "                         const C11& postpad_w)\n"
            "        : image_(image), filter_h_(filter_h(0)), filter_w_(filter_h(0)),\n"
            "          stride_h_(stride_h(0)), stride_w_(stride_w(0)),\n"
            "          dilate_h_(dilate_h(0)), dilate_w_(dilate_w(0)),\n"
            "          prepad_h_(prepad_h(0)), prepad_w_(prepad_w(0)),\n"
            "          postpad_h_(postpad_h(0)), postpad_w_(postpad_w(0)),\n"
            "          i_shape_(image_.shape()),\n"
            "          o_height_(\n"
            "              (i_shape_[", h_dim, "] + prepad_h_ + postpad_h_ - (dilate_h_ * (filter_h_ - 1) + 1)) /\n"
            "              stride_h_ + 1\n"
            "          ),\n"
            "          o_width_(\n"
            "              (i_shape_[", w_dim, "] + prepad_w_ + postpad_w_ - (dilate_w_ * (filter_w_ - 1) + 1)) /\n"
            "              stride_w_ + 1\n"
            "          ),\n"
            "          shape_({\n"
            "              filter_h_ * filter_w_ * i_shape_[", channel_dim, "],\n"
            "              o_height_ * o_width_ * i_shape_[", batch_dim, "]\n"
            "          }) {}\n"
            "    XINLINE const Shape<ndim>& shape() const {\n"
            "        return shape_;\n"
            "    }\n"
            "    XINLINE T operator[](const Shape<ndim>& query) {\n"
            "        const int c = query[0] % i_shape_[", channel_dim, "];\n"
            "        const int n = query[1] / (o_height_ * o_width_);\n"
            "        const int i_without_channels = query[0] / i_shape_[", channel_dim, "];\n"
            "        const int w_offset = i_without_channels % filter_w_ * dilate_w_;\n"
            "        const int idivp    = i_without_channels / filter_w_;\n"
            "        const int h_offset = idivp % filter_h_ * dilate_h_;\n"
            "        const int w = (query[1] % o_width_) * stride_w_ + w_offset - prepad_w_;\n"
            "        const int jdivw = query[1] / o_width_;\n"
            "        const int h = (jdivw % o_height_) * stride_h_ + h_offset - prepad_h_;\n"
            "        if (0 <= w && w < i_shape_[", w_dim, "] &&\n"
            "            0 <= h && h < i_shape_[", h_dim, "]) {\n"
            "            return image_", access_image, ";\n"
            "        } else {\n"
            "            // padding with zeros:\n"
            "            return T(0.0f);\n"
            "        }\n"
            "    }\n"
            "};\n",
            template_string,
            "Im2ColKernel", data_format_, template_usage," im2col_kernel(\n"
            "     const C1& a,\n"
            "     const C2& b,\n"
            "     const C3& c,\n"
            "     const C4& d,\n"
            "     const C5& e,\n"
            "     const C6& f,\n"
            "     const C7& g,\n"
            "     const C8& h,\n"
            "     const C9& i,\n"
            "     const C10& j,\n"
            "     const C11& k) {\n"
            "    return Im2ColKernel", data_format_, template_usage, "(a, b, c, d, e, f, g, h, i, j, k);\n"
            "}\n"
        );
    }

    std::vector<int> bshape() const {
        return deduce_im2col_shape(image_->bshape(),
                                   filter_h_,
                                   filter_w_,
                                   stride_h_,
                                   stride_w_,
                                   dilate_h_,
                                   dilate_w_,
                                   prepad_h_,
                                   prepad_w_,
                                   postpad_h_,
                                   postpad_w_,
                                   data_format_);
    }

    virtual int ndim() const {
        return 2;
    }

    bool is_dim_collapsible_with_dim_minus_one(const int& dim) const {
        return false;
    }

    operation_state_ptr collapse_dim_with_dim_minus_one(const int& dim) const {
        throw std::runtime_error(
            "Cannot collapse dim with dim minus one result of im2col."
        );
        return shared_from_this();
    }

    operation_state_ptr transpose(const std::vector<int>& permutation) const {
        throw std::runtime_error(
            "Cannot transpose result of im2col."
        );
        return shared_from_this();
    }

    void compute_node_compilation_info(
            int desired_computation_rank,
            const std::vector<int>& desired_computation_shape,
            std::vector<const ArrayOperationState*>* arrays,
            std::vector<const ScalarOperationState*>* scalars,
            node_to_info_t* node_to_info) const {
        (*node_to_info)[this].computation_rank = desired_computation_rank;
        image_->compute_node_compilation_info(4, image_->shape(), arrays, scalars, node_to_info);
        filter_h_op_->compute_node_compilation_info(1, {}, arrays, scalars, node_to_info);
        filter_w_op_->compute_node_compilation_info(1, {}, arrays, scalars, node_to_info);
        stride_h_op_->compute_node_compilation_info(1, {}, arrays, scalars, node_to_info);
        stride_w_op_->compute_node_compilation_info(1, {}, arrays, scalars, node_to_info);
        dilate_h_op_->compute_node_compilation_info(1, {}, arrays, scalars, node_to_info);
        dilate_w_op_->compute_node_compilation_info(1, {}, arrays, scalars, node_to_info);
        prepad_h_op_->compute_node_compilation_info(1, {}, arrays, scalars, node_to_info);
        prepad_w_op_->compute_node_compilation_info(1, {}, arrays, scalars, node_to_info);
        postpad_h_op_->compute_node_compilation_info(1, {}, arrays, scalars, node_to_info);
        postpad_w_op_->compute_node_compilation_info(1, {}, arrays, scalars, node_to_info);
        (*node_to_info)[this].hash = utils::Hasher().add(optype_hash)
                                                    .add(desired_computation_rank)
                                                    .add(data_format_)
                                                    .add(node_to_info->at(image_.get()).hash)
                                                    .value();
    }

    std::string get_call_code_nd(
            const symbol_table_t& symbol_table,
            const node_to_info_t& node_to_info) const {
        return utils::make_message("im2col_kernel(",
                                    image_->get_call_code_nd(symbol_table, node_to_info),
                                    ",",
                                    filter_h_op_->get_call_code_nd(symbol_table, node_to_info),
                                    ",",
                                    filter_w_op_->get_call_code_nd(symbol_table, node_to_info),
                                    ",",
                                    stride_h_op_->get_call_code_nd(symbol_table, node_to_info),
                                    ",",
                                    stride_w_op_->get_call_code_nd(symbol_table, node_to_info),
                                    ",",
                                    dilate_h_op_->get_call_code_nd(symbol_table, node_to_info),
                                    ",",
                                    dilate_w_op_->get_call_code_nd(symbol_table, node_to_info),
                                    ",",
                                    prepad_h_op_->get_call_code_nd(symbol_table, node_to_info),
                                    ",",
                                    prepad_w_op_->get_call_code_nd(symbol_table, node_to_info),
                                    ",",
                                    postpad_h_op_->get_call_code_nd(symbol_table, node_to_info),
                                    ",",
                                    postpad_w_op_->get_call_code_nd(symbol_table, node_to_info),
                                    ")");
    }
};

const hash_t Im2ColOperationState::optype_hash = std::hash<std::string>()("Im2ColOperationState");

namespace op {
    Operation im2col(const Operation& image,
                     int filter_h,
                     int filter_w,
                     int stride_h,
                     int stride_w,
                     const std::string& data_format) {
        int image_ndim = image.ndim();
        ASSERT2(image_ndim == 3 || image_ndim == 4, utils::make_message(
            "im2col takes an image with ndim == 3 or ndim == 4 (got ndim=",
            image_ndim, ")."));
        ASSERT2(data_format.size() == 4, utils::make_message("data_format"
            " should be 4 character string containing letters N, C, H and W ("
            "got ", data_format, ")."));
        ASSERT2(data_format.find('N') != -1, utils::make_message("data_format"
            " should contain character 'N' (got ", data_format, ")."));
        ASSERT2(data_format.find('C') != -1, utils::make_message("data_format"
            " should contain character 'C' (got ", data_format, ")."));
        int h_dim = data_format.find('H');
        ASSERT2(h_dim != -1, utils::make_message("data_format"
            " should contain character 'H' (got ", data_format, ")."));
        int w_dim = data_format.find('W');
        ASSERT2(w_dim != -1, utils::make_message("data_format"
            " should contain character 'W' (got ", data_format, ")."));
        // No N dimension if image is 3D:
        if (image_ndim == 3) {
            w_dim = w_dim - 1;
            h_dim = h_dim - 1;
        }
        auto image_bshape = image.bshape();
        const int image_w = image_bshape[w_dim];
        const int image_h = image_bshape[h_dim];
        bool image_w_gt_patch = image_w >= filter_w;
        bool image_h_gt_patch = image_h >= filter_h;
        ASSERT2(image_w_gt_patch && image_h_gt_patch, utils::make_message("the "
            "image shape of im2col with data_format=", data_format, " should be "
            "smaller than filter size (filter_h=", filter_h, " vs. image_h=",
            image_h, ", filter_w=", filter_w, " vs. w_dim=", image_w, ")."));

        return Operation(
            std::make_shared<Im2ColOperationState>(
                image.state_,
                filter_h,
                filter_w,
                stride_h,
                stride_w,
                /*dilate_h=*/1,
                /*dilate_w=*/1,
                /*prepad_h=*/0,
                /*prepad_w=*/0,
                /*pospad_h=*/0,
                /*pospad_w=*/0,
                data_format
            )
        );
    }
}  // namespace op
