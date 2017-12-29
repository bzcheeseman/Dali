#include "outer.h"
#include "dali/utils/make_message.h"
#include "dali/utils/assert2.h"
#include "dali/array/jit/jit_runner.h"
#include "dali/array/jit/jit_utils.h"
#include "dali/array/op/elementwise_operation.h"
namespace op {
    namespace jit {
        struct Outer : public JITNode {
            static const hash_t optype_hash;
            Array left_, right_;
            Outer(Array left, Array right) : JITNode(2, {left.shape()[0], right.shape()[0]}, left.dtype()),
                left_(left), right_(right) {}

            std::string kernel_name(const node_to_info_t& node_to_info) const {
                return utils::make_message("outer", node_to_info.at(this).computation_rank, "d");
            }

            std::string prefix_code(const node_to_info_t& node_to_info, memory::DeviceT device_type) const {
                return define_kernel(/*ndim=*/node_to_info.at(this).computation_rank,
                                     /*has_shape=*/true,
                                     /*arguments=*/{"left", "right"},
                                     /*kernel=*/"left_[query[ndim - 2]] * right_[query[ndim - 1]]",
                                     /*name=*/kernel_name(node_to_info),
                                     /*is_assignable=*/false);
            }

            std::vector<Array> arguments() const {return {left_, right_};}
            expression_ptr copy() const {return std::make_shared<Outer>(left_, right_);}

            virtual void compute_node_compilation_info(int desired_computation_rank,
                                                       const std::vector<int>& desired_computation_shape,
                                                       SymbolTable& symbol_table,
                                                       node_to_info_t* node_to_info) const {
                (*node_to_info)[this].computation_rank = desired_computation_rank;
                (*node_to_info)[this].computation_shape = desired_computation_shape;
                symbol_table.declare_shape(this);
                op::jit::compute_node_compilation_info(left_,
                    1, {desired_computation_shape[desired_computation_shape.size() - 2]},
                    symbol_table, node_to_info);
                op::jit::compute_node_compilation_info(right_,
                    1, {desired_computation_shape[desired_computation_shape.size() - 1]},
                    symbol_table, node_to_info);
                (*node_to_info)[this].hash = utils::Hasher().add(optype_hash)
                                                            .add(desired_computation_rank)
                                                            .add(node_to_info->at(left_.expression().get()).hash)
                                                            .add(node_to_info->at(right_.expression().get()).hash)
                                                            .value();
            }

            std::string get_call_code_nd(
                    const SymbolTable& symbol_table,
                    const node_to_info_t& node_to_info,
                    memory::DeviceT device_type) const {
                return utils::make_message(kernel_name(node_to_info), "(",
                                           op::jit::get_call_code_nd(left_, symbol_table, node_to_info, device_type),
                                           ",",
                                           op::jit::get_call_code_nd(right_, symbol_table, node_to_info, device_type),
                                           ",",
                                           symbol_table.get_shape(this),
                                           ")");
            }
        };
        const hash_t Outer::optype_hash = std::hash<std::string>()(typeid(Outer).name());
    }  // namespace jit

    Array outer(Array a, Array b) {
        std::tie(a, b) = ensure_arguments_compatible(a.ravel(), b.ravel(), "outer");
        return Array(std::make_shared<jit::Outer>(a, b));
    }
}  // namespace op
