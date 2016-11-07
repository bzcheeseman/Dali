#include "expression.h"

#include "dali/array/op2/expression/array_wrapper.h"
#include "dali/array/op2/expression/abstract_assign.h"
#include "dali/array/op2/rtc/rtc_expression.h"
#include "dali/array/op2/rtc/scalar_wrapper.h"

namespace expression {

    std::vector<int> ExpressionState::shape() const {
        std::vector<int> res = bshape();
        std::transform(res.begin(), res.end(), res.begin(),
            [](const int& x) {return std::abs(x);}
        );
        return res;
    }

    int ExpressionState::ndim() const {
        return bshape().size();
    }

    int ExpressionState::number_of_elements() const {
        return hypercube_volume(shape());
    }

    void ExpressionState::full_operation_name(std::stringstream* ss) const {
        *ss << name();
        auto args = arguments();
        if (args.size() > 0) {
            *ss << "(";
            for (int i = 0; i < args.size(); i++) {
                args[i]->full_operation_name(ss);
                if (i + 1 != args.size()) {
                    *ss << ", ";
                }
            }
            *ss << ")";
        }
    }

    std::string ExpressionState::full_operation_name() const {
        std::stringstream ss;
        full_operation_name(&ss);
        return ss.str();
    }


    std::vector<std::shared_ptr<const ExpressionState>> ExpressionState::arguments() const {
        return {};
    }


    // TODO(jonathan, szymon): This explicitly locates array ops, it's a bit of a violation
    // of contract because, everywhere else, we assume that compute_compilation_info is the
    // one responsible for identifying the arrays participating in the expression.
    // Hence, we should refactor the code so that we only collect args in one place.

    // returns device_proposal, device_found (if not args are present it's hard to suggest anything)
    std::tuple<memory::Device, bool> ExpressionState::preferred_device() const {
        int args_read = 0;
        bool shared_common_device = true;
        memory::Device common_preferred_device;
        memory::Device output_device;

        for_all_suboperations([&](const ExpressionState* op) {
            auto ptr = dynamic_cast<const ArrayWrapper*>(op);


            if (ptr != NULL) {
                auto mem = ptr->array_.memory();

                // When state args_read <= 0, then reduction is in its first Array argument
                // while other non-Array arguments have been ignored by ReduceOverArgs<>::reduce_helper
                // [Note: output is also an Array argument]
                if (args_read <= 0) {
                    // *** When considering the first Array ***
                    // If there's only 1 Array involved, we can safely consider
                    // this Array's memory's preferred_device as a good option
                    output_device = mem->preferred_device;
                    // One caveat, we want preferred_device's memory to be fresh
                    bool is_best_option_fresh = mem->is_fresh(mem->preferred_device);
                    // Also we want to know whether any copy of memory is fresh
                    bool is_some_other_option_fresh = mem->is_any_fresh();
                    // if the preferred memory is not fresh, and there is
                    // a fresh alternative use it:
                    if (!is_best_option_fresh && is_some_other_option_fresh) {
                        output_device = mem->find_some_fresh_device();
                    }// else, make the preferred device fresh

                    common_preferred_device = mem->preferred_device;
                } else {
                    if (mem->preferred_device != common_preferred_device || !shared_common_device) {
                        // When considering other arguments, if the next argument prefers a different device,
                        // then we fallback to the tie-breaker device
                        output_device = memory::default_preferred_device;
                        shared_common_device = false;
                    } else {
                        // we can place the computation on the currently agreed device
                    }
                }
                ++args_read;
            }
        });

        if (args_read == 0) {
            return std::make_tuple(memory::default_preferred_device, false);
        } else {
            return std::make_tuple(output_device, true);
        }

    }

    ExpressionState::operator Assignable<Array> () const {
        auto this_ptr = shared_from_this();
        return Assignable<Array>([this_ptr](Array& out, const OPERATOR_T& operator_t) mutable {
            // auto output_dtype_proposal  = this_ptr->dtype();
            // auto output_bshape_proposal = this_ptr->bshape();

            // bool device_found;
            // memory::Device output_device_proposal;
            // std::tie(output_device_proposal, device_found) = this_ptr->preferred_device();

            // auto op = Expression(this_ptr);
            // Array out_array;


            // OPERATOR_T operator_to_use = operator_t == OPERATOR_T_LSE ? OPERATOR_T_ADD : operator_t;

            // if (operator_t == OPERATOR_T_LSE) {
            //     std::vector<int> reduction_axes = get_auto_reduce_axes(out, output_bshape_proposal);
            //     if (reduction_axes.size() > 0) {
            //         op = op::sum(op, reduction_axes);
            //         // add the reduced dimensions back:
            //         for (int i = 0; i < reduction_axes.size(); ++i) {
            //             output_bshape_proposal[reduction_axes[i]] = 1;
            //         }
            //     }
            //     initialize_output_array(
            //         out,
            //         output_dtype_proposal,
            //         output_device_proposal,
            //         &output_bshape_proposal
            //     );
            //     out_array = out;
            //     for (int i = int(reduction_axes.size()) - 1; i >= 0; --i) {
            //         out_array = out_array.squeeze(reduction_axes[i]);
            //     }
            //     if (reduction_axes.size() > 0) {
            //         output_bshape_proposal = op.bshape();
            //     }
            // } else {
            //     initialize_output_array(
            //         out,
            //         output_dtype_proposal,
            //         output_device_proposal,
            //         &output_bshape_proposal
            //     );
            //     out_array = out;
            // }

            // TODO(szymon): infer the device
            // auto output_device = memory::Device::cpu();
            // TODO(szymon): ensure out is initialized
            auto self_op = op::assign(out, operator_t, Expression(this_ptr));
            auto runnable_self_op = self_op.state_->as_rvalue()->as_runnable(self_op.preferred_device());

            // Currently a stateless array in does not get modified by the initiliazation code of
            // assignment, so the result needs to be propagated back
            if (out.is_stateless()) {
                out = runnable_self_op->destination_op()->as_array()->array_;
            }
            // auto optimized_self_op = runnable_self_op.optimize();
            runnable_self_op->run_all();

            // eval_op(self_op, self_op.shape(), output_device);
        });
    }

    ExpressionState::operator Assignable<ArrayGather> () const {
        auto this_ptr = shared_from_this();
        return Assignable<ArrayGather>([this_ptr](ArrayGather& out, const OPERATOR_T& operator_t) mutable {
            // auto output_dtype  = out.dtype();
            // auto output_device = memory::Device::cpu();
            // auto self_op = op::assign(op::gather(out.source, out.indices), operator_t == OPERATOR_T_LSE ? OPERATOR_T_ADD : operator_t, Expression(this_ptr));
            // eval_op(self_op, self_op.shape(), output_device);
        });
    }

    ExpressionState::operator Assignable<ArraySubtensor> () const {
        auto this_ptr = shared_from_this();
        return Assignable<ArraySubtensor>([this_ptr](ArraySubtensor& out, const OPERATOR_T& operator_t) mutable {
            // auto output_dtype  = out.dtype();
            // auto output_device = memory::Device::cpu();
            // auto self_op = op::assign(op::gather_from_rows(out.source, out.indices), operator_t == OPERATOR_T_LSE ? OPERATOR_T_ADD : operator_t, Expression(this_ptr));
            // eval_op(self_op, self_op.shape(), output_device);
        });
    }

    void ExpressionState::for_all_suboperations(std::function<void(const ExpressionState*)> callback) const {
        callback(this);
        for (auto& child: arguments()) {
            child->for_all_suboperations(callback);
        }
    }


    std::shared_ptr<const LValue> ExpressionState::as_lvalue() const {
        return std::dynamic_pointer_cast<const LValue>(shared_from_this());
    }

    std::shared_ptr<const RValue> ExpressionState::as_rvalue() const {
        return std::dynamic_pointer_cast<const RValue>(shared_from_this());
    }

    std::shared_ptr<const rtc::RtcExpression> ExpressionState::as_jit() const {
        // TODO(jonathan): ensure that for non-jit operation we have a sensible way of representing them.
        return std::dynamic_pointer_cast<const rtc::RtcExpression>(shared_from_this());
    }


    std::shared_ptr<const ArrayWrapper> ExpressionState::as_array() const {
        // TODO(jonathan): ensure that for non-jit operation we have a sensible way of representing them.
        return std::dynamic_pointer_cast<const ArrayWrapper>(shared_from_this());
    }


    ///////////////////////////////////////////////////////////////////////////////
    //                         RVALUE                                           //
    ///////////////////////////////////////////////////////////////////////////////

    std::shared_ptr<const Runnable> RValue::add_to(std::shared_ptr<const LValue> op, memory::Device device) const {
        ASSERT2(false, "not implemented, awaiting jit"); // TODO(szymon): implement A += Assign(temp, op);
    }
    std::shared_ptr<const Runnable> RValue::sub_to(std::shared_ptr<const LValue> op, memory::Device device) const {
        ASSERT2(false, "not implemented, awaiting jit"); // TODO(szymon): implement A -= Assign(temp, op);
    }
    std::shared_ptr<const Runnable> RValue::mul_to(std::shared_ptr<const LValue> op, memory::Device device) const {
        ASSERT2(false, "not implemented, awaiting jit"); // TODO(szymon): implement A *= Assign(temp, op);
    }
    std::shared_ptr<const Runnable> RValue::div_to(std::shared_ptr<const LValue> op, memory::Device device) const {
        ASSERT2(false, "not implemented, awaiting jit"); // TODO(szymon): implement A /= Assign(temp, op);
    }

    std::shared_ptr<const Runnable> RValue::as_runnable(memory::Device device) const {
        return std::make_shared<const AbstractAssign>(
                initialize_destination(device)->as_lvalue(),
                OPERATOR_T_EQL,
                shared_from_this()->as_rvalue()
        )->as_runnable(device);
    }

    std::shared_ptr<const ArrayWrapper>    RValue::initialize_destination(memory::Device device) const {
        Array temp(shape(), dtype(), device);
        return std::make_shared<const ArrayWrapper>(temp);
    }

    std::shared_ptr<const Runnable> RValue::operator_to(
            OPERATOR_T operator_t,
            std::shared_ptr<const LValue> op,
            memory::Device device) const {
        if (operator_t == OPERATOR_T_EQL) {
            return assign_to(op, device);
        } else if (operator_t == OPERATOR_T_ADD) {
            return add_to(op, device);
        } else if (operator_t == OPERATOR_T_SUB) {
            return sub_to(op, device);
        } else if (operator_t == OPERATOR_T_MUL) {
            return mul_to(op, device);
        } else if (operator_t == OPERATOR_T_DIV) {
            return div_to(op, device);
        } else {
            ASSERT2(false, "unexpected operator_t");
        }
    }

    ///////////////////////////////////////////////////////////////////////////////
    //                         LVALUE                                           //
    ///////////////////////////////////////////////////////////////////////////////

    std::shared_ptr<const Runnable> LValue::operator_from(
            OPERATOR_T operator_t,
            std::shared_ptr<const Runnable> op,
            memory::Device device) const {
        if (operator_t == OPERATOR_T_EQL) {
            return assign_from(op, device);
        } else if (operator_t == OPERATOR_T_ADD) {
            return add_from(op, device);
        } else if (operator_t == OPERATOR_T_SUB) {
            return sub_from(op, device);
        } else if (operator_t == OPERATOR_T_MUL) {
            return mul_from(op, device);
        } else if (operator_t == OPERATOR_T_DIV) {
            return div_from(op, device);
        } else {
            ASSERT2(false, "unexpected operator_t");
        }
    }

    ///////////////////////////////////////////////////////////////////////////////
    //                         LRVALUE                                           //
    ///////////////////////////////////////////////////////////////////////////////

    std::shared_ptr<const Runnable> LRValue::add_from(std::shared_ptr<const Runnable> op, memory::Device device) const {
        ASSERT2(false, "not implemented, awaiting jit"); // TODO(szymon): implement A = A + op;
    }

    std::shared_ptr<const Runnable> LRValue::sub_from(std::shared_ptr<const Runnable> op, memory::Device device) const {
        ASSERT2(false, "not implemented, awaiting jit"); // TODO(szymon): implement A = A - op;
    }

    std::shared_ptr<const Runnable> LRValue::mul_from(std::shared_ptr<const Runnable> op, memory::Device device) const {
        ASSERT2(false, "not implemented, awaiting jit"); // TODO(szymon): implement A = A * op;
    }

    std::shared_ptr<const Runnable> LRValue::div_from(std::shared_ptr<const Runnable> op, memory::Device device) const {
        ASSERT2(false, "not implemented, awaiting jit"); // TODO(szymon): implement A = A / op;
    }

    ///////////////////////////////////////////////////////////////////////////////
    //                         RUNNABLE                                          //
    ///////////////////////////////////////////////////////////////////////////////

    std::shared_ptr<const Runnable> Runnable::assign_to(std::shared_ptr<const LValue> op, memory::Device device) const {
        auto dest_rvalue = destination_op()->as_rvalue();
        ASSERT2(dest_rvalue, "Runnable can only be interpreted as RValue if destination_op is an rvalue.");
        dest_rvalue->assign_to(op, device);
    }

    std::shared_ptr<const Runnable> Runnable::add_to(std::shared_ptr<const LValue> op, memory::Device device) const {
        auto dest_rvalue = destination_op()->as_rvalue();
        ASSERT2(dest_rvalue, "Runnable can only be interpreted as RValue if destination_op is an rvalue.");
        dest_rvalue->add_to(op, device);
    }

    std::shared_ptr<const Runnable> Runnable::sub_to(std::shared_ptr<const LValue> op, memory::Device device) const {
        auto dest_rvalue = destination_op()->as_rvalue();
        ASSERT2(dest_rvalue, "Runnable can only be interpreted as RValue if destination_op is an rvalue.");
        dest_rvalue->sub_to(op, device);
    }

    std::shared_ptr<const Runnable> Runnable::mul_to(std::shared_ptr<const LValue> op, memory::Device device) const {
        auto dest_rvalue = destination_op()->as_rvalue();
        ASSERT2(dest_rvalue, "Runnable can only be interpreted as RValue if destination_op is an rvalue.");
        dest_rvalue->mul_to(op, device);
    }

    std::shared_ptr<const Runnable> Runnable::div_to(std::shared_ptr<const LValue> op, memory::Device device) const {
        auto dest_rvalue = destination_op()->as_rvalue();
        ASSERT2(dest_rvalue, "Runnable can only be interpreted as RValue if destination_op is an rvalue.");
        dest_rvalue->div_to(op, device);
    }

    void Runnable::run_all() const {
        for (const auto& arg : arguments()) {
            auto arg_runnable = std::dynamic_pointer_cast<const Runnable>(arg);
            if (arg_runnable) arg_runnable->run_all();
        }
        run();
    }

    ///////////////////////////////////////////////////////////////////////////////
    //                         EXPRESSION                                        //
    ///////////////////////////////////////////////////////////////////////////////

    std::string Expression::name() const {
        return state_->full_operation_name();
    }

    Expression::Expression(const Array& arr): Expression(std::make_shared<ArrayWrapper>(arr)) {
    }

    Expression::Expression(const Assignable<Array>& arr): Expression(std::make_shared<ArrayWrapper>(Array(arr))) {
    }

    Expression::Expression(double scalar): Expression(std::make_shared<rtc::ScalarWrapperDouble>(scalar)) {
    }

    Expression::Expression(int scalar): Expression(std::make_shared<rtc::ScalarWrapperInteger>(scalar)) {
    }

    Expression::Expression(float scalar): Expression(std::make_shared<rtc::ScalarWrapperFloat>(scalar)) {
    }

    Expression::Expression(std::shared_ptr<const ExpressionState> state): state_(state) {
    }

    DType Expression::dtype() const {
        return state_->dtype();
    }

    int Expression::ndim() const {
        return state_->ndim();
    }

    memory::Device Expression::preferred_device() const {
        return std::get<0>(state_->preferred_device());
    }

    std::vector<int> Expression::bshape() const {
        return state_->bshape();
    }

    std::vector<int> Expression::shape() const {
        return state_->shape();
    }

    bool Expression::is_scalar() const {
        return ndim() == 0;
    }

    int Expression::number_of_elements() const {
        return state_->number_of_elements();
    }

    Expression::operator Assignable<Array> () const {
        return state_->operator Assignable<Array>();
    }
    Expression::operator Assignable<ArrayGather> () const {
        return state_->operator Assignable<ArrayGather>();
    }
    Expression::operator Assignable<ArraySubtensor> () const {
        return state_->operator Assignable<ArraySubtensor>();
    }


}  // namespace expression
