#include "computation.h"
#include "dali/array/array.h"
#include "dali/array/expression/assignment.h"
#include "dali/array/expression/control_flow.h"

#include "dali/utils/make_message.h"
#include <unordered_map>
#include <algorithm>

Computation::Computation(Array left, OPERATOR_T operator_t, Array right, Array assignment) :
    left_(left), operator_t_(operator_t), right_(right), assignment_(assignment) {}

void Computation::run_and_cleanup() {
    run();
    auto buffer_expression = left_.expression()->copy();
    auto assignment_expression = assignment_.expression();
    buffer_expression->strides_ = assignment_expression->strides_;
    buffer_expression->shape_ = assignment_expression->shape_;
    buffer_expression->offset_ = assignment_expression->offset_;
    assignment_.set_expression(buffer_expression);

}

struct Noop : public Computation {
    using Computation::Computation;
    void run() {}
};

std::unordered_map<const char*, std::vector<to_computation_t> > IMPLEMENTATIONS;

void convert_array_to_ops(const Array& element,
                          std::vector<std::shared_ptr<Computation>>& steps,
                          std::vector<Array>& elements) {
    if (element.is_assignment()) {
        auto assignment = op::static_as_assignment(element);
        auto assignment_left = assignment->left();
        auto hashname = typeid(*assignment->right().expression()).name();
        bool found_impl = false;
        if (IMPLEMENTATIONS.find(hashname) != IMPLEMENTATIONS.end()) {
            for (const auto& impl_creator : IMPLEMENTATIONS[hashname]) {
                auto impl = impl_creator(assignment_left,
                                         assignment->operator_t_,
                                         assignment->right(),
                                         element);
                if (impl != nullptr) {
                    steps.emplace_back(impl);
                    found_impl = true;
                    break;
                }
            }
            if (found_impl) {
                // TODO(jonathan): this is a hack and should be removed
                if (assignment->right().is_assignment()) {
                    elements.emplace_back(assignment->right());
                } else {
                    auto args = assignment->right().expression()->arguments();
                    elements.insert(elements.end(), args.begin(), args.end());
                }
                elements.emplace_back(assignment->left());
            }
        }
        if (!found_impl) {
            throw std::runtime_error(utils::make_message(
                "No implementation found for ", assignment->right().expression_name(), "."));
        }
    } else if (element.is_control_flow()) {
        auto cflow = op::static_as_control_flow(element);
        auto conditions = cflow->conditions();
        // insert dummy node that turns ControlFlow node back into a BufferView
        // when all conditions are met.
        steps.emplace_back(std::make_shared<Noop>(
            cflow->left(), OPERATOR_T_EQL, element, element));
        elements.insert(elements.end(), conditions.begin(), conditions.end());
        elements.emplace_back(cflow->left());
    } else if (!element.is_assignable()) {
        throw std::runtime_error(utils::make_message(
            "Can only convert Assignments and Buffers "
            "to ops (got ", element.expression_name(), ")."));
    }
}

// TODO(jonathan): add this from Python
std::vector<std::shared_ptr<Computation>> convert_to_ops(Array root) {
    std::vector<std::shared_ptr<Computation>> steps;
    std::vector<Array> elements({root});
    while (!elements.empty()) {
        auto element = elements.back();
        elements.pop_back();
        convert_array_to_ops(element, steps, elements);
    }
    std::reverse(steps.begin(), steps.end());
    return steps;
}

int register_implementation(const char* opname, to_computation_t impl) {
    IMPLEMENTATIONS[opname].emplace_back(impl);
    return 0;
}
