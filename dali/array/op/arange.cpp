#include "arange.h"
#include "dali/array/jit/jit_runner.h"
#include "dali/array/jit/jit_utils.h"
#include "dali/utils/make_message.h"

namespace op {
namespace jit {

struct Arange : public JITNode {
    static const hash_t optype_hash;
    Arange(Array start, Array step, int size) :
        JITNode(1, {size}, start.dtype(), {start, step}) {
    }

    virtual memory::Device preferred_device() const {
        return arguments_[0].preferred_device();
    }
    virtual std::string get_call_code_nd(const SymbolTable& symbol_table,
                                         const node_to_info_t& node_to_info,
                                         memory::DeviceT device_type) const {
        return utils::make_message(
            kernel_name(node_to_info), "(",
            op::jit::get_call_code_nd(arguments_[0], symbol_table, node_to_info, device_type), ", ",
            op::jit::get_call_code_nd(arguments_[1], symbol_table, node_to_info, device_type), ", ",
            symbol_table.get_shape(this), ")");
    }

    virtual void compute_node_compilation_info(int desired_computation_rank,
                                               const std::vector<int>& desired_computation_shape,
                                               SymbolTable& symbol_table,
                                               node_to_info_t* node_to_info) const {
        (*node_to_info)[this].computation_rank = desired_computation_rank;
        (*node_to_info)[this].computation_shape = desired_computation_shape;
        symbol_table.declare_shape(this);
        op::jit::compute_node_compilation_info(arguments_[0],
                                               1,
                                               {1},
                                               symbol_table,
                                               node_to_info);
        op::jit::compute_node_compilation_info(arguments_[1],
                                               1,
                                               {1},
                                               symbol_table,
                                               node_to_info);
        utils::Hasher hasher;
        hasher.add(optype_hash)
              .add(desired_computation_rank)
              .add(node_to_info->at(arguments_[0].expression().get()).hash)
              .add(node_to_info->at(arguments_[1].expression().get()).hash);
        (*node_to_info)[this].hash = hasher.value();
    }

    virtual std::string kernel_name(const node_to_info_t& node_to_info) const {
        return utils::make_message("arange", node_to_info.at(this).computation_rank, "d");
    }

    virtual std::string prefix_code(const node_to_info_t& node_to_info,
                                    memory::DeviceT device_type) const {
        return define_kernel(/*ndim=*/node_to_info.at(this).computation_rank,
                             /*has_shape=*/true,
                             /*arguments=*/{"start", "step"},
                             /*kernel=*/"start_[0] + indices_to_offset(shape_, query) * step_[0]",
                             /*name=*/kernel_name(node_to_info),
                             /*is_assignable=*/false);
    }

    virtual expression_ptr copy() const {
        return std::make_shared<Arange>(arguments_[0], arguments_[1], shape_[0]);
    }
};

const hash_t Arange::optype_hash = std::hash<std::string>()(typeid(Arange).name());

}  // namespace jit

Array arange(int size) {
    return arange(0, size, 1);
}

Array arange(int start, int stop) {
    return arange(start, stop, 1);
}

Array arange(int start, int stop, int step) {
    ASSERT2(start != stop, utils::make_message(
        "arange's start and stop must be different "
        "(got start = ", start, ", stop = ", stop, ")."));
    int size = std::max((stop - start) / step, 1);
    ASSERT2(size > 0, utils::make_message(
        "arange's size must be strictly positive "
        "(got start = ", start, ", stop = ", stop,
        ", step = ", step, ")."));
    return Array(std::make_shared<jit::Arange>(start, step, size));
}

}  // namespace op
