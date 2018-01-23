#include "circular_convolution.h"
#include "dali/array/jit/jit_runner.h"
#include "dali/array/jit/jit_utils.h"
#include "dali/array/op/elementwise_operation.h"
#include "dali/utils/make_message.h"

namespace op {
    namespace jit {
        struct CircularConvolution : public JITNode {
            static const hash_t optype_hash;

            CircularConvolution(const Array& content, const Array& weights) :
                    JITNode(std::max(2, std::max(min_computation_rank(content), min_computation_rank(weights))),
                            get_common_shape({content, weights}),
                            content.dtype(), {content, weights}) {}

            expression_ptr copy() const {
                return std::make_shared<CircularConvolution>(arguments_[0], arguments_[1]);
            }

            std::string prefix_code(const node_to_info_t& node_to_info, memory::DeviceT device_type) const {
                std::string kernel = (
                    "T res = static_cast<T>(0);\n"
                    "const int conv_size = shape_[ndim - 1];\n"
                    "const int& x = query[ndim - 1];\n"
                    "Shape<ndim> content_query = query;\n"
                    "Shape<ndim> weights_query = query;\n"
                    "int& shift_idx = weights_query[ndim - 1];\n"
                    "int& offset = content_query[ndim - 1];\n"
                    "#pragma clang loop vectorize(enable)\n"
                    "#pragma clang loop interleave(enable)\n"
                    "for (shift_idx = 0; shift_idx < conv_size; shift_idx++) {\n"
                    "    offset = x + shift_idx;\n"
                    "    if (offset >= conv_size) {\n"
                    "        offset -= conv_size;\n"
                    "    }\n"
                    "    res += content_[content_query] * weights_[weights_query];\n"
                    "}\n"
                    "return res;\n");
                return define_kernel(/*ndim=*/node_to_info.at(this).computation_rank,
                                     /*has_shape=*/true,
                                     /*arguments=*/{"content", "weights"},
                                     /*kernel=*/kernel,
                                     /*name=*/kernel_name(node_to_info),
                                     /*is_assignable=*/false);
            }

            bool is_axis_collapsible_with_axis_minus_one(int axis) const {
                if (axis == ndim() - 1) {
                    return false;
                }
                return arguments_[0].is_axis_collapsible_with_axis_minus_one(axis) &&
                       arguments_[1].is_axis_collapsible_with_axis_minus_one(axis);
            }

            expression_ptr collapse_axis_with_axis_minus_one(int axis) const {
                return std::make_shared<CircularConvolution>(
                    arguments_[0].collapse_axis_with_axis_minus_one(axis),
                    arguments_[1].collapse_axis_with_axis_minus_one(axis));
            }

            void compute_node_compilation_info(
                    int desired_computation_rank,
                    const std::vector<int>& desired_computation_shape,
                    SymbolTable& symbol_table,
                    node_to_info_t& node_to_info) const {
                node_to_info[this].computation_rank = desired_computation_rank;
                op::jit::compute_node_compilation_info(arguments_[0], desired_computation_rank, desired_computation_shape, symbol_table, node_to_info);
                op::jit::compute_node_compilation_info(arguments_[1], desired_computation_rank, desired_computation_shape, symbol_table, node_to_info);
                node_to_info[this].hash = utils::Hasher().add(optype_hash)
                                                            .add(desired_computation_rank)
                                                            .add(node_to_info.at(arguments_[0].expression().get()).hash)
                                                            .add(node_to_info.at(arguments_[1].expression().get()).hash)
                                                            .value();

            }

            virtual bool shape_required() const {return true;}

            std::string kernel_name(const node_to_info_t& node_to_info) const {
                return utils::make_message("circular_convolution_", node_to_info.at(this).computation_rank, "d");
            }

            std::string get_call_code_nd(
                    const SymbolTable& symbol_table,
                    const node_to_info_t& node_to_info,
                    memory::DeviceT device_type) const {
                return generate_call_code_nd(this,
                                             kernel_name(node_to_info),
                                             symbol_table, node_to_info, device_type,
                                             /*has_shape=*/true);
            }
        };
        const hash_t CircularConvolution::optype_hash = std::hash<std::string>()(typeid(CircularConvolution).name());
    } // namespace jit
    Array circular_convolution(Array x, Array weights) {
        std::tie(x, weights) = ensure_arguments_compatible(x, weights, "circular_convolution", true);
        return Array(std::make_shared<jit::CircularConvolution>(x, weights));
    }
}  // namespace op
