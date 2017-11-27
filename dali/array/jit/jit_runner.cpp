#include "jit_runner.h"
#include "dali/utils/make_message.h"
#include "dali/utils/assert2.h"

namespace op {
namespace jit {

// CONVENIENCE METHODS //
std::shared_ptr<JITNode> as_jit_node(Array array) {
    auto casted = std::dynamic_pointer_cast<JITNode>(array.expression());
    ASSERT2(casted != nullptr, utils::make_message("Attempting to cast a non-jit node expression "
        "into a jit node."));
    return casted;
}

hash_t node_hash(const node_to_info_t& node_to_info, const Array& array) {
    return node_to_info.at(array.expression().get()).hash;
}

// JIT NODE

JITNode::JITNode(int min_computation_rank,
                 const std::vector<int>& shape,
                 DType dtype,
                 int offset,
                 const std::vector<int>& strides) : Expression(shape, dtype, offset, strides),
                                                    min_computation_rank_(min_computation_rank) {}

JITNode::JITNode(const JITNode& other) : Expression(other),
                                         min_computation_rank_(other.min_computation_rank_) {};


std::string JITNode::prefix_code(const node_to_info_t& node_to_info, memory::DeviceT device_type) const {
    return "";
}

bool JITNode::is_axis_collapsible_with_axis_minus_one(const int& axis) const {
    return false;
}

memory::Device JITNode::preferred_device() const {
    memory::Device best_device = memory::Device::device_of_doom();
    // TODO(jonathan): ensure this logic actually picks the right device
    // to run based on the inputs, not just agreement
    for (auto& arg : arguments()) {
        auto new_pref_device = arg.preferred_device();
        if (best_device.is_error()) {
            best_device = new_pref_device;
        } else {
            if (new_pref_device == best_device) {
                best_device = new_pref_device;
            } else {
                best_device = memory::default_preferred_device;
                break;
            }
        }
    }
    return best_device;
}

// JIT RUNNER //

JITRunner::JITRunner(Array root, const std::vector<Array>& leaves) :
        JITNode(root.ndim(), root.shape(), root.dtype()),
        root_(root), leaves_(leaves) {
    if (std::dynamic_pointer_cast<JITRunner>(root.expression())) {
        throw std::runtime_error("JITRunner should not contain a JITRunner.");
    }
}

std::vector<Array> JITRunner::arguments() const {
    return leaves_;
}
// TODO(jonathan): add pretty-printing here to keep track of what was jitted or not.

expression_ptr JITRunner::copy() const {
    return std::make_shared<JITRunner>(*this);
}

memory::Device JITRunner::preferred_device() const {
    return root_.preferred_device();
}

bool JITRunner::is_axis_collapsible_with_axis_minus_one(const int& axis) const {
    return as_jit_node(root_)->is_axis_collapsible_with_axis_minus_one(axis);
}

void JITRunner::compute_node_compilation_info(int desired_computation_rank,
                                           const std::vector<int>& desired_computation_shape,
                                           std::vector<std::shared_ptr<BufferView>>* arrays,
                                           std::vector<std::shared_ptr<ScalarView>>* scalars,
                                           node_to_info_t* node_to_info) const {
    throw std::runtime_error("should not be called from the JITRunner.");
}

std::string JITRunner::get_call_code_nd(const symbol_table_t& symbol_table,
                                        const node_to_info_t& node_to_info,
                                        memory::DeviceT device_type) const {
    throw std::runtime_error("should not be called from the JITRunner.");
}

}  // namespace jit
}  // namespace op
