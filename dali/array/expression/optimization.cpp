#include "optimization.h"
#include "dali/array/expression/assignment.h"
#include "dali/utils/make_message.h"

std::vector<Array> right_args(Array node) {
    auto buffer = std::dynamic_pointer_cast<Assignment>(node.expression());
    return buffer->right_.expression()->arguments();
}

namespace {

    struct Optimization {
        std::function<bool(const Array&)> condition_;
        std::function<Array(const Array&)> transformation_;
        std::string name_;
        bool matches(const Array& array) const {
            return condition_(array);
        }
        Array transform(const Array& array) const {
            return transformation_(array);
        }
        Optimization(std::function<bool(const Array&)> condition,
                     std::function<Array(const Array&)> transformation,
                     const std::string& name) :
            condition_(condition), transformation_(transformation), name_(name) {}
    };

    // TODO(jonathan): add this from Python
    Array all_assignments_or_buffers(Array node) {
        if (node.is_buffer()) {
            return node;
        }
        if (!node.is_assignment()) {
            node.set_expression(op::to_assignment(node).expression());
        }
        Assignment* node_assign = static_cast<Assignment*>(node.expression().get());
        if (node_assign->right_.is_assignment()) {
            Assignment* node_right_assign = op::static_as_assignment(node_assign->right_);
            if (node_right_assign->operator_t_ == OPERATOR_T_EQL &&
                node_right_assign->right_.expression()->supports_operator(node_assign->operator_t_)) {
                node_assign->right_.set_expression(node_right_assign->right_.expression());
            }
        }
        for (auto& arg : right_args(node)) {
            arg.set_expression(all_assignments_or_buffers(arg).expression());
        }
        return node;
    }

    std::vector<Optimization> OPTIMIZATIONS;
    std::vector<std::string> OPTIMIZATION_NAMES;

    Array simplify_destination(Array root) {
        // leaf node:
        if (root.is_buffer()) {
            return root;
        }
        // recurse on children:
        std::vector<Array> children;
        if (root.is_assignment()) {
            children.emplace_back(op::static_as_assignment(root)->right_);
        } else {
            children = root.expression()->arguments();
        }

        // recurse on arguments of node:
        for (auto& arg : children) {
            arg.set_expression(simplify_destination(arg).expression());
        }
        for (const auto& optimization : OPTIMIZATIONS) {
            if (optimization.matches(root)) {
                auto new_root = optimization.transform(root);
                // guard optimization behavior so that shapes are preserved
                ASSERT2(new_root.shape() == root.shape(), utils::make_message(
                    "Optimization '", optimization.name_, "' altered the shape of the operation"
                    " from ", root.shape(), " to ", new_root.shape(),
                    " on expression ", root.full_expression_name(), "."
                    ));
                ASSERT2(new_root.dtype() == root.dtype(), utils::make_message(
                    "Optimization '", optimization.name_, "' altered the dtype of the operation"
                    " from ", dtype_to_name(root.dtype()), " to ", dtype_to_name(new_root.dtype()),
                    " on expression ", root.full_expression_name(), "."
                    ));
                root = new_root;
            }
        }
        return root;
    }
}

int register_optimization(std::function<bool(const Array&)> condition,
                          std::function<Array(const Array&)> transformation,
                          const std::string& name) {
    OPTIMIZATIONS.emplace_back(condition, transformation, name);
    return 0;
}

Array canonical(const Array& array) {
    // assignment pass
    auto node = all_assignments_or_buffers(array);

    // simplification pass (jit, merge, etc...)
    return simplify_destination(node);
}

