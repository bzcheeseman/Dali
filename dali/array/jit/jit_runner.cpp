#include "jit_runner.h"
#include "dali/utils/make_message.h"
#include "dali/utils/assert2.h"
#include "dali/array/expression/optimization.h"
#include "dali/array/expression/assignment.h"
#include "dali/array/op/binary.h"

namespace op {
namespace jit {

// CONVENIENCE METHODS //
std::shared_ptr<JITNode> as_jit_node(Array array) {
    auto casted = std::dynamic_pointer_cast<JITNode>(array.expression());
    ASSERT2(casted != nullptr, utils::make_message("Attempting to cast a non-jit node expression (",
        typeid(*array.expression()).name(), ") into a jit node."));
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
                                              std::vector<const BufferView*>* arrays,
                                              std::vector<const ScalarView*>* scalars,
                                              node_to_info_t* node_to_info) const {
    throw std::runtime_error("should not be called from the JITRunner.");
}

std::string JITRunner::get_call_code_nd(const symbol_table_t& symbol_table,
                                        const node_to_info_t& node_to_info,
                                        memory::DeviceT device_type) const {
    throw std::runtime_error("should not be called from the JITRunner.");
}

bool is_jit_node(const Array& array) {
    auto node = std::dynamic_pointer_cast<JITNode>(array.expression());
    return node != nullptr;
}

bool is_jit_runner(const Array& array) {
    auto node = std::dynamic_pointer_cast<JITRunner>(array.expression());
    return node != nullptr;
}

bool is_jit_assignment(const Array& node) {
    return (node.is_assignment() &&
            is_jit_node(as_assignment(node)->right_) &&
            !is_jit_runner(as_assignment(node)->right_));
}

std::shared_ptr<JITRunner> as_jit_runner(const Array& array) {
    return std::dynamic_pointer_cast<JITRunner>(array.expression());
}

Array jit_root(const Array& array) {
    if (is_jit_runner(array)) {
        return as_jit_runner(array)->root_;
    }
    return array;
}

std::tuple<Array, Array> replace_assign_with_inplace(const Array& node) {
    auto assign = as_assignment(node);
    auto rightside = jit_root(assign->right_);
    auto operator_t = assign->operator_t_;
    if (operator_t == OPERATOR_T_EQL) {
        return std::tuple<Array, Array>(rightside, Array());
    } else if (operator_t == OPERATOR_T_ADD) {
        return std::tuple<Array, Array>(op::add(assign->left_, rightside), assign->left_);
    } else if (operator_t == OPERATOR_T_SUB) {
        return std::tuple<Array, Array>(op::subtract(assign->left_, rightside), assign->left_);
    } else if (operator_t == OPERATOR_T_MUL) {
        return std::tuple<Array, Array>(op::eltmul(assign->left_, rightside), assign->left_);
    } else if (operator_t == OPERATOR_T_DIV) {
        return std::tuple<Array, Array>(op::eltdiv(assign->left_, rightside), assign->left_);
    } else {
        throw std::runtime_error(utils::make_message("No way to replace_assign_with_inplace using operator ",
                                                     operator_to_name(operator_t), "."));
    }
}

Array jit_merge(const Array& root) {
    std::vector<Array> leaves;
    auto assign = as_assignment(root);
    auto root_buffer = assign->left_;
    auto root_operator = assign->operator_t_;
    Array left_leaf, replaced;
    for (auto& arg : right_args(root)) {
        if (arg.is_assignment() &&
            is_jit_runner(as_assignment(arg)->right_)) {
            // grab leaves from existing jit-runner recursively:
            auto extra_leaves = as_jit_runner(as_assignment(arg)->right_)->leaves_;
            leaves.insert(leaves.end(), extra_leaves.begin(), extra_leaves.end());
            // if the node is an assignment to a buffer, ensure that
            // the assignment op gets included within this op
            // (e.g. by spoofing the assignment and replacing it with
            //  the equivalent JIT op)
            std::tie(replaced, left_leaf) = replace_assign_with_inplace(arg);
            // if the assignment involves using the left-side (e.g.
            // left += right -> left + right), then keep the left node
            // as a dependency leaf:
            if (!left_leaf.is_stateless()) {
                leaves.emplace_back(left_leaf);
            }
            // now that the jitrunners and assignments are gone, connect
            // up the new operation in the graph:
            arg.set_expression(replaced.expression());
        } else {
            // this node is either an assignment, or a buffer,
            // and is needed as an input here:
            leaves.emplace_back(arg);
        }
    }

    auto new_root = assign->right_;
    return Array(std::make_shared<Assignment>(
        // keep the original target buffer:
        root_buffer, root_operator,
        // use the merged operation instead
        Array(std::make_shared<JITRunner>(new_root, leaves))));
}

int registered = register_optimization(is_jit_assignment, jit_merge);

}  // namespace jit
}  // namespace op
