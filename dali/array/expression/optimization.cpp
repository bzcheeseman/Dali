#include "optimization.h"
#include "dali/array/expression/assignment.h"

std::vector<Array> right_args(Array node) {
    auto buffer = std::dynamic_pointer_cast<Assignment>(node.expression());
    return buffer->right_.expression()->arguments();
}

namespace {
    Array autoreduce_assign(Array left, Array right) {
        throw std::runtime_error("autoreduce_assign not implemented yet.");
    }

    Array assign(Array left, OPERATOR_T operator_t, Array right);

    Array to_assignment(Array node) {
        return assign(Array::zeros(node.shape(), node.dtype()),
                      OPERATOR_T_EQL,
                      Array(node.expression()));
    }

    Array assign(Array left, OPERATOR_T operator_t, Array right) {
        if (operator_t == OPERATOR_T_EQL) {
            return Array(std::make_shared<Assignment>(left, operator_t, right));
        } else if (operator_t == OPERATOR_T_LSE) {
            return autoreduce_assign(left, right);
        } else {
            // a temp is added so that non overwriting operators
            // can be run independently from the right side's evaluation.
            return Array(std::make_shared<Assignment>(left, operator_t, to_assignment(right)));
        }
    }

    struct Optimization {
        std::function<bool(const Array&)> condition_;
        std::function<Array(const Array&)> transformation_;
        bool matches(const Array& array) const {
            return condition_(array);
        }
        Array transform(const Array& array) const {
            return transformation_(array);
        }
        Optimization(std::function<bool(const Array&)> condition,
                     std::function<Array(const Array&)> transformation) :
            condition_(condition), transformation_(transformation) {}
    };

    // TODO(jonathan): add this from Python
    Array all_assignments_or_buffers(Array node) {
        if (node.is_buffer()) {
            return node;
        }
        if (!node.is_assignment()) {
            node.set_expression(to_assignment(node).expression());
        }
        for (auto& arg : right_args(node)) {
            arg.set_expression(all_assignments_or_buffers(arg).expression());
        }
        return node;
    }

    std::vector<Optimization> OPTIMIZATIONS;

    Array simplify_destination(Array root) {
        // leaf node:
        if (root.is_buffer()) {
            return root;
        }
        // recurse on children:
        std::vector<Array> children;
        if (root.is_assignment()) {
            children.emplace_back(std::dynamic_pointer_cast<Assignment>(root.expression())->right_);
        } else {
            children = root.expression()->arguments();
        }

        // recurse on arguments of node:
        for (auto& arg : children) {
            arg.set_expression(simplify_destination(arg).expression());
        }
        for (const auto& optimization : OPTIMIZATIONS) {
            if (optimization.matches(root)) {
                root = optimization.transform(root);
            }
        }
        return root;
    }
}

int register_optimization(std::function<bool(const Array&)> condition,
                          std::function<Array(const Array&)> transformation) {
    OPTIMIZATIONS.emplace_back(condition, transformation);
    return 0;
}

Array canonical(const Array& array) {
    // assignment pass
    auto node = all_assignments_or_buffers(array);

    // simplification pass (jit, merge, etc...)
    return simplify_destination(node);
}

