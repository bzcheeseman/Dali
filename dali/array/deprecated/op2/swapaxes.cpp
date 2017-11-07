#include "swapaxes.h"

#include "dali/array/op2/rtc/rtc_expression.h"
#include "dali/array/op2/rtc/scalar_wrapper.h"

namespace expression {
namespace rtc {
struct SwapaxesExpressionNode : public RtcExpression {
    static const hash_t optype_hash;

    std::shared_ptr<const RtcExpression> input_;
    int axis1_;
    int axis2_;
    std::shared_ptr<const ScalarWrapperInteger> axis1_operation_;
    std::shared_ptr<const ScalarWrapperInteger> axis2_operation_;

    SwapaxesExpressionNode(std::shared_ptr<const RtcExpression> input,
                            int axis1,
                            int axis2) :
            RtcExpression(input->ndim()),
            input_(input),
            axis1_(axis1),
            axis2_(axis2),
            axis1_operation_(std::make_shared<ScalarWrapperInteger>(axis1)),
            axis2_operation_(std::make_shared<ScalarWrapperInteger>(axis2)) {
    }

    virtual std::string name() const {
        return "swapaxes";
    }

    std::string kernel_identifier(const node_to_info_t& node_to_info) const {
        int comp_rank = node_to_info.at(this).computation_rank;
        return utils::make_message(
            comp_rank, "D", axis1_, "to", axis2_
        );
    }

    std::string prefix_code(const node_to_info_t& node_to_info, memory::DeviceT device_type) const {
        std::stringstream ss;
        int comp_rank = node_to_info.at(this).computation_rank;
        for (int i = 0; i < comp_rank; i++) {
            int axis = i;
            if (axis == axis1_) {
                axis = axis2_;
            } else if (axis == axis2_) {
                axis = axis1_;
            }
            ss << "query[" << axis << "]";
            if (i + 1 != comp_rank) {
                ss << ", ";
            }
        }
        auto query_remapping = ss.str();
        auto class_id = kernel_identifier(node_to_info);

        return utils::make_message("template<typename C1, typename C2, typename C3>\n"
        "struct SwapaxesKernel", class_id," {\n"
        "    const C1 input_;\n"
        "    const C2 axis1_;\n"
        "    const C3 axis2_;\n"
        "    static const int ndim = C1::ndim;\n"
        "    typedef typename C1::T T;\n"
        "    XINLINE SwapaxesKernel", class_id, "(const C1& indices, const C2& axis1, const C3& axis2)\n"
        "        : input_(indices), axis1_(axis1), axis2_(axis2) {}\n"
        "    XINLINE T operator[](Shape<ndim> query) {\n"
        "        return input_[{", query_remapping, "}];\n"
        "    }\n"
        "};\n"
        "template<typename C1, typename C2, typename C3>\n"
        "SwapaxesKernel", class_id, "<C1, C2, C3> swapaxes_kernel_", class_id, "(const C1& a, const C2& b, const C3& c) {\n"
        "    return SwapaxesKernel", class_id, "<C1, C2, C3>(a, b, c);\n"
        "}\n");
    }

    DType dtype() const {
        return input_->dtype();
    }

    std::vector<int> bshape() const {
        auto result = input_->bshape();
        std::swap(result[axis1_], result[axis2_]);
        return result;
    }

    std::vector<std::shared_ptr<const ExpressionNode>> arguments() const {
        return {input_, axis1_operation_, axis2_operation_};
    }

    bool is_dim_collapsible_with_dim_minus_one(const int& dim) const {
        return false;
    }

    std::shared_ptr<const RtcExpression> collapse_dim_with_dim_minus_one(const int& dim) const {
        return std::make_shared<SwapaxesExpressionNode>(
            input_->collapse_dim_with_dim_minus_one(dim),
            axis1_, axis2_
        );
    }

    std::shared_ptr<const RtcExpression> transpose(const std::vector<int>& permutation) const {
        auto new_permutation = permutation;
        // cumulate swapaxes with the transpose:
        std::swap(new_permutation[axis1_], new_permutation[axis2_]);
        return input_->transpose(new_permutation);
    }

    void compute_node_compilation_info(
            int desired_computation_rank,
            const std::vector<int>& desired_computation_shape,
            std::vector<const RtcArrayWrapper*>* arrays,
            std::vector<const ScalarWrapper*>* scalars,
            node_to_info_t* node_to_info) const {
        (*node_to_info)[this].computation_rank = desired_computation_rank;
        auto input_shape = desired_computation_shape;
        // send shape to input with transformation removed
        std::swap(input_shape[axis1_], input_shape[axis2_]);
        input_->compute_node_compilation_info(desired_computation_rank, input_shape, arrays, scalars, node_to_info);
        axis1_operation_->compute_node_compilation_info(1, {}, arrays, scalars, node_to_info);
        axis2_operation_->compute_node_compilation_info(1, {}, arrays, scalars, node_to_info);
        (*node_to_info)[this].hash = utils::Hasher().add(optype_hash)
                                                    .add(desired_computation_rank)
                                                    .add(node_to_info->at(input_.get()).hash)
                                                    // ensure that the axes are part of the hash:
                                                    .add(axis1_)
                                                    .add(axis2_)
                                                    .add(node_to_info->at(axis1_operation_.get()).hash)
                                                    .add(node_to_info->at(axis2_operation_.get()).hash)
                                                    .value();
    }


    std::string get_call_code_nd(
            const symbol_table_t& symbol_table,
            const node_to_info_t& node_to_info,
            memory::DeviceT device_type) const {
        auto class_id = kernel_identifier(node_to_info);
        return utils::make_message("swapaxes_kernel_", class_id, "(",
                                    input_->get_call_code_nd(symbol_table, node_to_info, device_type),
                                    ",",
                                    axis1_operation_->get_call_code_nd(symbol_table, node_to_info, device_type),
                                    ",",
                                    axis2_operation_->get_call_code_nd(symbol_table, node_to_info, device_type),
                                    ")");
    }
};
const hash_t SwapaxesExpressionNode::optype_hash = std::hash<std::string>()("SwapaxesExpressionNode");
}  // namespace rtc
}  // namespace expression

namespace op {
    expression::ExpressionGraph swapaxes(const expression::ExpressionGraph& input, int axis1, int axis2) {
        int input_ndim = input.ndim();
        if (input_ndim == 0) return input;
        if (axis1 < 0) axis1 = input_ndim + axis1;
        if (axis2 < 0) axis2 = input_ndim + axis2;
        // no-op
        if (axis1 == axis2) return input;

        ASSERT2(0 <= axis1 && axis1 < input_ndim, utils::make_message("swapaxes"
            " axis1 (", axis1, ") must be less than ndim (", input_ndim, ")."));
        ASSERT2(0 <= axis2 && axis2 < input_ndim, utils::make_message("swapaxes"
            " axis2 (", axis2, ") must be less than ndim (", input_ndim, ")."));

        if (auto input_as_array = input.state_->as_array()) {
            return expression::ExpressionGraph(
                input_as_array->array_.swapaxes(axis1, axis2)
            );
        } else {
            auto input_as_jit = input.state_->as_jit();
            if (!input_as_jit) {
                auto input_as_rvalue = input.state_->as_rvalue();
                ASSERT2(input_as_rvalue, "input to swapaxes must be an RValue.");
                auto input_as_runnable = input_as_rvalue->as_runnable(input.preferred_device());
                input_as_jit = input_as_runnable->destination_op()->as_jit();
            }
            return expression::ExpressionGraph(
                std::make_shared<expression::rtc::SwapaxesExpressionNode>(
                    input_as_jit,
                    // ensure order is canonicalized (to avoid creating 2 ops
                    // when one is sufficient)
                    std::min(axis1, axis2),
                    std::max(axis1, axis2)
                )
            );
        }
    }
}  // namespace
