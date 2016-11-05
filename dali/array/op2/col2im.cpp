#include "col2im.h"

#include <map>

#include "dali/array/op2/expression/expression.h"
#include "dali/array/op2/rtc/rtc_expression.h"
#include "dali/array/op2/rtc/scalar_wrapper.h"
#include "dali/array/op2/elementwise_operation.h"
#include "dali/array/op2/rtc_utils.h"
#include "dali/array/op/spatial/utils.h"
#include "dali/utils/hash_utils.h"
#include "dali/utils/make_message.h"

inline int get_image_dim(const std::vector<int>& image_shape, const std::string& data_format, char c) {
    if (image_shape.size() == 3) {
        // no n dimension
        return image_shape[data_format.find(c) - 1];
    } else {
        // 4 dimensions map to data_format adequately
        return image_shape[data_format.find(c)];
    }
}
namespace expression {
namespace rtc {

struct Col2ImExpressionState : public RtcExpression {
    static const hash_t optype_hash;
    std::shared_ptr<const RtcExpression> input_;
    std::vector<int> image_shape_;

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

    std::shared_ptr<const ScalarWrapperInteger> filter_h_op_;
    std::shared_ptr<const ScalarWrapperInteger> filter_w_op_;

    std::shared_ptr<const ScalarWrapperInteger> stride_h_op_;
    std::shared_ptr<const ScalarWrapperInteger> stride_w_op_;

    std::shared_ptr<const ScalarWrapperInteger> dilate_h_op_;
    std::shared_ptr<const ScalarWrapperInteger> dilate_w_op_;

    std::shared_ptr<const ScalarWrapperInteger> prepad_h_op_;
    std::shared_ptr<const ScalarWrapperInteger> prepad_w_op_;

    std::shared_ptr<const ScalarWrapperInteger> o_height_op_;
    std::shared_ptr<const ScalarWrapperInteger> o_width_op_;
    std::shared_ptr<const ScalarWrapperInteger> o_channel_op_;

    Col2ImExpressionState(std::shared_ptr<const RtcExpression> input,
                         const std::vector<int>& image_shape,
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
            RtcExpression(image_shape.size()),
            input_(input),
            image_shape_(image_shape),
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
            filter_h_op_(std::make_shared<ScalarWrapperInteger>(filter_h)),
            filter_w_op_(std::make_shared<ScalarWrapperInteger>(filter_w)),
            stride_h_op_(std::make_shared<ScalarWrapperInteger>(stride_h)),
            stride_w_op_(std::make_shared<ScalarWrapperInteger>(stride_w)),
            dilate_h_op_(std::make_shared<ScalarWrapperInteger>(dilate_h)),
            dilate_w_op_(std::make_shared<ScalarWrapperInteger>(dilate_w)),
            prepad_h_op_(std::make_shared<ScalarWrapperInteger>(prepad_h)),
            prepad_w_op_(std::make_shared<ScalarWrapperInteger>(prepad_w)),
            o_height_op_(
                std::make_shared<ScalarWrapperInteger>((get_image_dim(image_shape, data_format, 'H')+ prepad_h_ + postpad_h_ - (dilate_h_ * (filter_h_ - 1) + 1)) / stride_h_ + 1)
            ),
            o_width_op_(
                std::make_shared<ScalarWrapperInteger>((get_image_dim(image_shape, data_format, 'W') + prepad_w_ + postpad_w_ - (dilate_w_ * (filter_w_ - 1) + 1)) / stride_w_ + 1)
            ),
            o_channel_op_(
                std::make_shared<ScalarWrapperInteger>(get_image_dim(image_shape, data_format, 'C'))
            ) {}

    virtual DType dtype() const {
        return input_->dtype();
    }

    virtual std::string name() const {
        return "col2im";
    }

    virtual std::vector<std::shared_ptr<const ExpressionState>> arguments() const {
        return {
            input_,
            filter_h_op_,
            filter_w_op_,
            stride_h_op_,
            stride_w_op_,
            dilate_h_op_,
            dilate_w_op_,
            prepad_h_op_,
            prepad_w_op_,
            o_height_op_,
            o_width_op_,
            o_channel_op_
        };
    }

    std::string prefix_code(const node_to_info_t& node_to_info, memory::DeviceT device_type) const {
        int computation_rank = node_to_info.at(this).computation_rank;

        std::string template_string =
            "template<typename C1, typename C2,\n"
            "typename C3, typename C4, typename C5,\n"
            "typename C6, typename C7, typename C8,\n"
            "typename C9, typename C10, typename C11,\n"
            "typename C12>\n";
        std::string template_usage =
            "<C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11, C12>";
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
            {'W', "((w - pw * stride_w_) / dilate_w_)"},
            {'H', "((h - ph * stride_h_) / dilate_h_)"}
        };
        std::map<unsigned char, std::string> letter2stride = {
            {'C', "o_channel_"},
            {'W', "filter_w_"},
            {'H', "filter_h_"}
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

        return utils::make_message(
            "XINLINE int Col2ImKernel", data_format_, "_min(int left, int right) {\n"
            "    return left < right ? left : right;\n"
            "}\n",
            template_string,
            "struct Col2ImKernel", data_format_, " {\n"
            "    const C1 input_;\n"
            "    const int filter_h_;\n"
            "    const int filter_w_;\n"
            "    const int stride_h_;\n"
            "    const int stride_w_;\n"
            "    const int dilate_h_;\n"
            "    const int dilate_w_;\n"
            "    const int prepad_h_;\n"
            "    const int prepad_w_;\n"
            "    const int o_height_;\n"
            "    const int o_width_;\n"
            "    const int o_channel_;\n"
            "    static const int ndim = ", computation_rank, ";\n"
            "    typedef typename C1::T T;\n"
            "    XINLINE Col2ImKernel", data_format_, "(\n"
            "                         const C1& input,\n"
            "                         const C2& filter_h,\n"
            "                         const C3& filter_w,\n"
            "                         const C4& stride_h,\n"
            "                         const C5& stride_w,\n"
            "                         const C6& dilate_h,\n"
            "                         const C7& dilate_w,\n"
            "                         const C8& prepad_h,\n"
            "                         const C9& prepad_w,\n"
            "                         const C10& o_height,\n"
            "                         const C11& o_width,\n"
            "                         const C12& o_channel)\n"
            "        : input_(input), filter_h_(filter_h(0)), filter_w_(filter_h(0)),\n"
            "          stride_h_(stride_h(0)), stride_w_(stride_w(0)),\n"
            "          dilate_h_(dilate_h(0)), dilate_w_(dilate_w(0)),\n"
            "          prepad_h_(prepad_h(0)), prepad_w_(prepad_w(0)),\n"
            "          o_height_(o_height(0)), o_width_(o_width(0)), o_channel_(o_channel(0)) {}\n"
            "    XINLINE T operator[](const Shape<ndim>& query) {\n"
            "        \n"
            "        const int c = query[", c_dim, "];\n"
            "        int w = query[", w_dim, "];\n"
            "        int h = query[", h_dim, "];\n"
            "        const int n = ", computation_rank == 3 ? "1" : utils::make_message("query[", n_dim, "]"), ";\n"
            "        \n"
            "        h += prepad_h_;\n"
            "        w += prepad_w_;\n"
            "        \n"
            "        const int patch_size_h_dilate = (dilate_h_ * (filter_h_ - 1) + 1);\n"
            "        const int patch_size_w_dilate = (dilate_w_ * (filter_w_ - 1) + 1);\n"
            "        \n"
            "        const int ph_min =\n"
            "            h < patch_size_h_dilate ? h % dilate_h_ : (h - patch_size_h_dilate + stride_h_) / stride_h_;\n"
            "        \n"
            "        const int pw_min =\n"
            "            w < patch_size_w_dilate ? w % dilate_w_ : (w - patch_size_w_dilate + stride_w_) / stride_w_;\n"
            "        const int ph_max = Col2ImKernel", data_format_, "_min((h + stride_h_) / stride_h_, o_height_);\n"
            "        const int pw_max = Col2ImKernel", data_format_, "_min((w + stride_w_) / stride_w_, o_width_);\n"
            "        T res = static_cast<T>(0);\n"
            "        for (int ph = ph_min; ph < ph_max; ph += dilate_h_) {\n"
            "            for (int pw = pw_min; pw < pw_max; pw += dilate_w_) {\n"
            "                res += input_[{\n",
                                access_im2coled_data, ", (n * o_height_ + ph) * o_width_ + pw\n"
            "                }];\n"
            "            }\n"
            "        }\n"
            "        return res;\n"
            "    }\n"
            "};\n",
            template_string,
            "Col2ImKernel", data_format_, template_usage," im2col_kernel(\n"
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
            "     const C11& k,\n"
            "     const C12& l) {\n"
            "    return Col2ImKernel", data_format_, template_usage, "(a, b, c, d, e, f, g, h, i, j, k, l);\n"
            "}\n"
        );
    }

    std::vector<int> bshape() const {
        return image_shape_;
    }
    std::vector<int> shape() const {
        return image_shape_;
    }

    virtual int ndim() const {
        return image_shape_.size();
    }

    bool is_dim_collapsible_with_dim_minus_one(const int& dim) const {
        return false;
    }

    std::shared_ptr<const RtcExpression> collapse_dim_with_dim_minus_one(const int& dim) const {
        throw std::runtime_error(
            "Cannot collapse dim with dim minus one result of col2im."
        );
        return jit_shared_from_this();
    }

    std::shared_ptr<const RtcExpression> transpose(const std::vector<int>& permutation) const {
        throw std::runtime_error(
            "Cannot transpose result of col2im."
        );
        return jit_shared_from_this();
    }

    void compute_node_compilation_info(
            int desired_computation_rank,
            const std::vector<int>& desired_computation_shape,
            std::vector<const RtcArrayWrapper*>* arrays,
            std::vector<const ScalarWrapper*>* scalars,
            node_to_info_t* node_to_info) const {
        (*node_to_info)[this].computation_rank = desired_computation_rank;
        input_->compute_node_compilation_info(2, input_->shape(), arrays, scalars, node_to_info);
        filter_h_op_->compute_node_compilation_info(1, {}, arrays, scalars, node_to_info);
        filter_w_op_->compute_node_compilation_info(1, {}, arrays, scalars, node_to_info);
        stride_h_op_->compute_node_compilation_info(1, {}, arrays, scalars, node_to_info);
        stride_w_op_->compute_node_compilation_info(1, {}, arrays, scalars, node_to_info);
        dilate_h_op_->compute_node_compilation_info(1, {}, arrays, scalars, node_to_info);
        dilate_w_op_->compute_node_compilation_info(1, {}, arrays, scalars, node_to_info);
        prepad_h_op_->compute_node_compilation_info(1, {}, arrays, scalars, node_to_info);
        prepad_w_op_->compute_node_compilation_info(1, {}, arrays, scalars, node_to_info);
        o_height_op_->compute_node_compilation_info(1, {}, arrays, scalars, node_to_info);
        o_width_op_->compute_node_compilation_info(1, {}, arrays, scalars, node_to_info);
        o_channel_op_->compute_node_compilation_info(1, {}, arrays, scalars, node_to_info);
        (*node_to_info)[this].hash = utils::Hasher().add(optype_hash)
                                                    .add(desired_computation_rank)
                                                    .add(data_format_)
                                                    .add(node_to_info->at(input_.get()).hash)
                                                    .value();
    }

    std::string get_call_code_nd(
            const symbol_table_t& symbol_table,
            const node_to_info_t& node_to_info,
            memory::DeviceT device_type) const {
        return utils::make_message("im2col_kernel(",
                                    input_->get_call_code_nd(symbol_table, node_to_info, device_type),
                                    ",",
                                    filter_h_op_->get_call_code_nd(symbol_table, node_to_info, device_type),
                                    ",",
                                    filter_w_op_->get_call_code_nd(symbol_table, node_to_info, device_type),
                                    ",",
                                    stride_h_op_->get_call_code_nd(symbol_table, node_to_info, device_type),
                                    ",",
                                    stride_w_op_->get_call_code_nd(symbol_table, node_to_info, device_type),
                                    ",",
                                    dilate_h_op_->get_call_code_nd(symbol_table, node_to_info, device_type),
                                    ",",
                                    dilate_w_op_->get_call_code_nd(symbol_table, node_to_info, device_type),
                                    ",",
                                    prepad_h_op_->get_call_code_nd(symbol_table, node_to_info, device_type),
                                    ",",
                                    prepad_w_op_->get_call_code_nd(symbol_table, node_to_info, device_type),
                                    ",",
                                    o_height_op_->get_call_code_nd(symbol_table, node_to_info, device_type),
                                    ",",
                                    o_width_op_->get_call_code_nd(symbol_table, node_to_info, device_type),
                                    ",",
                                    o_channel_op_->get_call_code_nd(symbol_table, node_to_info, device_type),
                                    ")");
    }
};

const hash_t Col2ImExpressionState::optype_hash = std::hash<std::string>()("Col2ImExpressionState");

}  // namespace rtc
}  // namespace expression

namespace op {
    expression::Expression col2im(const expression::Expression& input,
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
        ASSERT2(data_format.size() == 4, utils::make_message("data_format"
            " should be 4 character string containing letters N, C, H and W ("
            "got ", data_format, ")."));
        int n_dim = data_format.find('N');
        ASSERT2(n_dim != -1, utils::make_message("data_format"
            " should contain character 'N' (got ", data_format, ")."));
        int c_dim = data_format.find('C');
        ASSERT2(c_dim != -1, utils::make_message("data_format"
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
            c_dim = c_dim - 1;
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

        return expression::Expression(
            std::make_shared<expression::rtc::Col2ImExpressionState>(
                input.state_->as_jit(),
                image_shape,
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
