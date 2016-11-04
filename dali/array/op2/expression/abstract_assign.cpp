#include "abstract_assign.h"

#include "dali/array/op2/rtc_utils.h"
#include "dali/array/op2/expression/array_wrapper.h"



namespace expression {
    std::vector<int> get_auto_reduce_axes(const Array& output, const std::vector<int>& in_bshape) {
        if (output.is_stateless() || output.ndim() != in_bshape.size()) {
            return {};
        }

        auto out_bshape = output.bshape();
        std::vector<int> reduction_axes;

        for (int i = 0; i < out_bshape.size(); ++i) {
            if (out_bshape[i] < 0) {
                ASSERT2(out_bshape[i] == -1,
                        "Assigning to broadcast_reshaped Array is not supported.");
                if (std::abs(in_bshape[i]) > 1) {
                    // see if number of reductions is greater than 1 or equal to size of output.
                    reduction_axes.emplace_back(i);
                }
            }
        }
        return reduction_axes;
    }

    std::shared_ptr<AbstractAssign> initialize_assignment_array(
            std::shared_ptr<const ArrayWrapper> left,
            const AbstractAssign* assignment,
            memory::Device device) {
        OPERATOR_T operator_to_use = assignment->operator_t_;
        Array out = left->array_;
        auto right = assignment->right_;
        auto output_dtype_proposal  = right->dtype();
        auto output_bshape_proposal = right->bshape();
        Array out_array;
        if (operator_to_use == OPERATOR_T_LSE) {
            // TODO: ensure that operator lse gets turned back into add:
            // OPERATOR_T operator_to_use = operator_t == OPERATOR_T_LSE ? OPERATOR_T_ADD : operator_t;
            ASSERT2(false, "LSE not yet supported (again).");
            // std::vector<int> reduction_axes = get_auto_reduce_axes(out, output_bshape_proposal);
            // if (reduction_axes.size() > 0) {
            //     op = op::sum(op, reduction_axes);
            //     // add the reduced dimensions back:
            //     for (int i = 0; i < reduction_axes.size(); ++i) {
            //         output_bshape_proposal[reduction_axes[i]] = 1;
            //     }
            // }
            // initialize_output_array(
            //     out,
            //     output_dtype_proposal,
            //     device,
            //     &output_bshape_proposal
            // );
            // out_array = out;
            // for (int i = int(reduction_axes.size()) - 1; i >= 0; --i) {
            //     out_array = out_array.squeeze(reduction_axes[i]);
            // }
            // if (reduction_axes.size() > 0) {
            //     output_bshape_proposal = op.bshape();
            // }
        } else {
            initialize_output_array(
                out,
                output_dtype_proposal,
                device,
                output_bshape_proposal
            );
            out_array = out;
        }

        if (!out.memory()->is_any_fresh() && operator_to_use == OPERATOR_T_ADD) {
            // if operation is += to an empty/zeros array, then switch operator
            // to equal:
            operator_to_use = OPERATOR_T_EQL;
        }
        return std::make_shared<AbstractAssign>(
            std::make_shared<ArrayWrapper>(out_array),
            operator_to_use,
            right
        );
    }



    AbstractAssign::AbstractAssign(
            std::shared_ptr<const LValue> left,
            const OPERATOR_T& operator_t,
            std::shared_ptr<const RValue> right) :
                    left_(left), right_(right), operator_t_(operator_t) {
    }

    DType AbstractAssign::dtype() const {
        return left_->dtype();
    }

    std::string AbstractAssign::name() const {
        return "assign";
    }

    void AbstractAssign::full_operation_name(std::stringstream* ss) const {
        left_->full_operation_name(ss);
        (*ss) << " " << operator_to_name(operator_t_) << " ";
        right_->full_operation_name(ss);
    }

    int AbstractAssign::ndim() const {
        return left_->ndim();
    }

    bool AbstractAssign::is_assignable() const {
        return true;
    }


    std::vector<int> AbstractAssign::bshape() const {
        return left_->bshape();
    }

    std::vector<std::shared_ptr<const ExpressionState>> AbstractAssign::arguments() const {
        return {left_, right_};
    }



    std::tuple<memory::Device, bool> AbstractAssign::preferred_device() const {
        if (left_->as_array()) {
            bool device_found;
            memory::Device output_device_proposal;
            std::tie(output_device_proposal, device_found) = right_->preferred_device();

            memory::Device output_device;
            const auto& out_array = left_->as_array()->array_;
            auto out_array_preferred_device = out_array.is_stateless() ?
                memory::default_preferred_device : out_array.memory()->preferred_device;
            if (device_found) {
                if (out_array_preferred_device != output_device_proposal) {
                    output_device = memory::default_preferred_device;
                    device_found = false;
                } else {
                    output_device = output_device_proposal;
                }
            } else {
                output_device = out_array_preferred_device;
            }
            return std::make_tuple(output_device, device_found);
        } else {
            return ((ExpressionState*)this)->preferred_device();
        }
    }


    std::shared_ptr<const Runnable> AbstractAssign::as_runnable(memory::Device device) const {
        if (left_->as_array() && left_->as_array()->array_.is_stateless()) {
            return initialize_assignment_array(
                left_->as_array(),
                this,
                device
            )->as_runnable(device);
        }

        if (operator_t_ == OPERATOR_T_EQL) {
            return right_->assign_to(left_, device);
        } else if (operator_t_ == OPERATOR_T_ADD) {
            return right_->add_to(left_, device);
        } else if (operator_t_ == OPERATOR_T_SUB) {
            return right_->sub_to(left_, device);
        } else if (operator_t_ == OPERATOR_T_MUL) {
            return right_->mul_to(left_, device);
        } else if (operator_t_ == OPERATOR_T_DIV) {
            return right_->div_to(left_, device);
        } else {
            ASSERT2(false, "not implemented.");
        }
    }

    std::shared_ptr<const ArrayWrapper> AbstractAssign::initialize_destination(memory::Device device) const {
        auto left_rvalue = left_->as_rvalue();
        ASSERT2(left_rvalue, "This assignment cannot be interpreted as rvalue.");
        return left_rvalue->initialize_destination(device);
    }

    std::shared_ptr<const Runnable> AbstractAssign::assign_to(std::shared_ptr<const LValue> op, memory::Device device) const {
        return as_runnable(device)->assign_to(op, device);
    }

    std::shared_ptr<const Runnable> AbstractAssign::add_to(std::shared_ptr<const LValue> op, memory::Device device) const {
        return as_runnable(device)->add_to(op, device);
    }

    std::shared_ptr<const Runnable> AbstractAssign::sub_to(std::shared_ptr<const LValue> op, memory::Device device) const {
        return as_runnable(device)->sub_to(op, device);
    }

    std::shared_ptr<const Runnable> AbstractAssign::mul_to(std::shared_ptr<const LValue> op, memory::Device device) const {
        return as_runnable(device)->mul_to(op, device);
    }

    std::shared_ptr<const Runnable> AbstractAssign::div_to(std::shared_ptr<const LValue> op, memory::Device device) const {
        return as_runnable(device)->div_to(op, device);
    }
}  // namespace expression



namespace op {
    expression::Expression assign(const expression::Expression& left, const OPERATOR_T& operator_t, const expression::Expression& right) {
        auto left_lvalue = left.state_->as_lvalue();
        auto right_rvalue = right.state_->as_rvalue();
        ASSERT2(left_lvalue, "Left side of assignment must be a lvalue.");
        ASSERT2(right_rvalue, "Right side of assignment must be a rvalue.");
        OPERATOR_T operator_to_use = operator_t;
        if (left.state_->as_array()) {
            ensure_output_array_compatible(left.state_->as_array()->array_, right.dtype(), right.bshape());
        }
        return expression::Expression(std::make_shared<expression::AbstractAssign>(left_lvalue, operator_t, right_rvalue));
    }
}
