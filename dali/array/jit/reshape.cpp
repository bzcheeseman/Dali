#include "reshape.h"
#include "dali/array/jit/jit_runner.h"
#include "dali/array/jit/jit_utils.h"
#include "dali/utils/make_message.h"

namespace op {
namespace jit {

struct Reshape : public JITNode {
    static const hash_t optype_hash;
    Reshape(Array array, const std::vector<int>& shape) :
        JITNode(min_computation_rank(array), shape, array.dtype(), {array}) {
    }

    virtual memory::Device preferred_device() const {
        return arguments_[0].preferred_device();
    }

    virtual std::string get_call_code_nd(const SymbolTable& symbol_table,
                                         const node_to_info_t& node_to_info,
                                         memory::DeviceT device_type) const {
        return generate_call_code_nd(this,
                                     kernel_name(node_to_info),
                                     symbol_table, node_to_info, device_type,
                                     /*has_shape=*/true);
    }

    virtual void compute_node_compilation_info(int desired_computation_rank,
                                               const std::vector<int>& desired_computation_shape,
                                               SymbolTable& symbol_table,
                                               node_to_info_t* node_to_info) const {
        (*node_to_info)[this].computation_rank = desired_computation_rank;
        (*node_to_info)[this].computation_shape = desired_computation_shape;
        symbol_table.declare_shape(this);
        op::jit::compute_node_compilation_info(arguments_[0],
                                               std::max(1, arguments_[0].ndim()),
                                               arguments_[0].ndim() == 0 ? std::vector<int>({1}) : arguments_[0].shape(),
                                               symbol_table,
                                               node_to_info);
        utils::Hasher hasher;
        hasher.add(optype_hash)
              .add(desired_computation_rank)
              .add(node_to_info->at(arguments_[0].expression().get()).hash);
        (*node_to_info)[this].hash = hasher.value();
    }

    virtual std::string kernel_name(const node_to_info_t& node_to_info) const {
        return utils::make_message("reshape", node_to_info.at(this).computation_rank, "d");
    }

    virtual std::string prefix_code(const node_to_info_t& node_to_info,
                                    memory::DeviceT device_type) const {
        return define_kernel(/*ndim=*/node_to_info.at(this).computation_rank,
                             /*has_shape=*/true,
                             /*arguments=*/{"array",},
                             /*kernel=*/"array_[index_to_dim(indices_to_offset(shape_, query), array_.shape())]",
                             /*name=*/kernel_name(node_to_info),
                             /*is_assignable=*/false);
    }

    virtual expression_ptr copy() const {
        return std::make_shared<Reshape>(arguments_[0], shape_);
    }
};

const hash_t Reshape::optype_hash = std::hash<std::string>()(typeid(Reshape).name());

Array jit_reshape(const Array& array, const std::vector<int>& shape) {
    return Array(std::make_shared<Reshape>(array, shape));
}
}  // namespace jit
}  // namespace op
