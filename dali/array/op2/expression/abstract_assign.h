#ifndef DALI_ARRAY_OP_EXPRESSION_ABSTRACT_ASSIGN_H
#define DALI_ARRAY_OP_EXPRESSION_ABSTRACT_ASSIGN_H

#include <memory>
#include <string>
#include <vector>

#include "dali/array/memory/device.h"
#include "dali/array/op2/expression/expression.h"

namespace expression {
    struct AbstractAssign : public RValue {
        std::shared_ptr<const LValue> left_;
        std::shared_ptr<const RValue> right_;
        OPERATOR_T operator_t_;

        AbstractAssign(std::shared_ptr<const LValue>  left,
                                     const OPERATOR_T& operator_t,
                                     std::shared_ptr<const RValue>  right);
        virtual DType dtype() const;

        virtual std::string name() const;

        virtual void full_operation_name(std::stringstream* ss) const;

        virtual bool is_assignable() const;
        virtual std::tuple<memory::Device, bool> preferred_device() const;

        virtual int ndim() const;

        virtual std::vector<int> bshape() const;

        virtual std::vector<std::shared_ptr<const ExpressionState>> arguments() const;

        virtual std::shared_ptr<const Runnable> assign_to(std::shared_ptr<const LValue> op, memory::Device device) const;
        virtual std::shared_ptr<const Runnable> add_to(std::shared_ptr<const LValue> op, memory::Device device) const;
        virtual std::shared_ptr<const Runnable> sub_to(std::shared_ptr<const LValue> op, memory::Device device) const;
        virtual std::shared_ptr<const Runnable> mul_to(std::shared_ptr<const LValue> op, memory::Device device) const;
        virtual std::shared_ptr<const Runnable> div_to(std::shared_ptr<const LValue> op, memory::Device device) const;


        virtual std::shared_ptr<const Runnable> as_runnable(memory::Device device) const;
        virtual std::shared_ptr<const ArrayWrapper>    initialize_destination(memory::Device device) const;
    };
}  // namespace expression


namespace op {
    expression::Expression assign(const expression::Expression& left, const OPERATOR_T& operator_t, const expression::Expression& right);
}  // namespace op

#endif  // DALI_ARRAY_OP_EXPRESSION_ABSTRACT_ASSIGN_H
