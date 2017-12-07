#include "computation.h"
#include "dali/array/array.h"
#include "dali/array/expression/assignment.h"
#include "dali/array/jit/jit_runner.h"

#include "dali/utils/make_message.h"
#include <unordered_map>

Computation::Computation(Array left, OPERATOR_T operator_t, Array right) :
    left_(left), operator_t_(operator_t), right_(right) {}


std::unordered_map<const char*, to_computation_t > IMPLEMENTATIONS;

struct Allocate : public Computation {
    Allocate(Array array) : Computation(array, OPERATOR_T_EQL, array) {}
    virtual void run() {}
};

// TODO(jonathan): add this from Python
std::vector<std::shared_ptr<Computation>> convert_to_ops(Array root) {
    std::vector<std::shared_ptr<Computation>> steps;
    std::vector<Array> elements({root});
    while (!elements.empty()) {
        auto element = elements.back();
        elements.pop_back();
        if (element.is_buffer()) {
            steps.emplace_back(std::make_shared<Allocate>(element));
        } else if (element.is_assignment()) {
            auto assignment = std::dynamic_pointer_cast<Assignment>(element.expression());
            if (assignment->right_.is_buffer() || assignment->right_.is_assignment()) {
                // # TODO(jonathan): clean this up
                element = op::jit::buffer_buffer_op(element);
            }
            auto hashname = typeid(*assignment->right_.expression()).name();
            if (IMPLEMENTATIONS.find(hashname) != IMPLEMENTATIONS.end()) {
                steps.emplace_back(
                    IMPLEMENTATIONS[hashname](assignment->right_,
                                              assignment->operator_t_,
                                              assignment->right_)
                );
                elements.emplace_back(assignment->left_);
                auto args = assignment->right_.expression()->arguments();
                elements.insert(elements.end(), args.begin(), args.end());
            } else {
                throw std::runtime_error(utils::make_message(
                    "No implementation found for ", assignment->right_.expression_name(), "."));
            }
        } else if (element.is_control_flow()) {
            auto args = element.expression()->arguments();
            elements.insert(elements.end(), args.begin(), args.end());
        } else {
            throw std::runtime_error(utils::make_message(
                "Can only convert Assignments and Buffers "
                "to ops (got ", element.expression_name(), ")."));
        }
    }
    std::reverse(steps.begin(), steps.end());
    return steps;
}

int register_implementation(const char* opname, to_computation_t impl) {
    // TODO(jonathan): this should be an incremental registry, should
    // not replace the previous version.
    IMPLEMENTATIONS[opname] = impl;
    return 0;
}