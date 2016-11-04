#ifndef DALI_ARRAY_OP_EXPRESSION_EXPRESSION_H
#define DALI_ARRAY_OP_EXPRESSION_EXPRESSION_H

#include <memory>
#include <string>
#include <vector>

#include "dali/array/dtype.h"
#include "dali/array/memory/device.h"
#include "dali/array/array.h"


namespace expression {
    struct LValue;
    struct RValue;
    struct Runnable;
    struct ArrayWrapper;
    namespace rtc {
        struct ScalarWrapper;
        struct RtcExpression;
    }

    struct ExpressionState : std::enable_shared_from_this<ExpressionState> {
        ///////////////////////////////////////////////////////////////////////////////
        //            MUST REIMPLEMENT FUNCTIONS BELOW                               //
        ///////////////////////////////////////////////////////////////////////////////
        virtual DType dtype() const = 0;
        virtual std::vector<int> bshape() const = 0;
        virtual std::string name() const = 0;

        ///////////////////////////////////////////////////////////////////////////////
        //            REIMPLEMENT AS YOU SEE FIT                                     //
        ///////////////////////////////////////////////////////////////////////////////

        virtual int ndim() const;

        virtual std::vector<int> shape() const;

        virtual int number_of_elements() const;

        virtual std::vector<std::shared_ptr<const ExpressionState>> arguments() const;

        // should almost never be reimplemented:
        virtual void full_operation_name(std::stringstream*) const;

        // returns device_proposal, device_found (if no args are present it's hard to suggest anything)
        virtual std::tuple<memory::Device, bool> preferred_device() const;

        ///////////////////////////////////////////////////////////////////////////////
        //            DO NOT REIMPLEMENT FUNCTIONS BELOW                             //
        ///////////////////////////////////////////////////////////////////////////////
        virtual std::string full_operation_name() const;

        virtual std::shared_ptr<const LValue>             as_lvalue()   const;
        virtual std::shared_ptr<const RValue>             as_rvalue()   const;
        virtual std::shared_ptr<const rtc::RtcExpression> as_jit()   const;
        virtual std::shared_ptr<const ArrayWrapper>       as_array()   const;

        operator Assignable<Array> () const;
        operator Assignable<ArrayGather> () const;
        operator Assignable<ArraySubtensor> () const;

        virtual void for_all_suboperations(std::function<void(const ExpressionState*)> callback) const final;
    };


    struct RValue: virtual public ExpressionState {
        virtual std::shared_ptr<const Runnable> assign_to(std::shared_ptr<const LValue> op, memory::Device device) const = 0;
        virtual std::shared_ptr<const Runnable> add_to(std::shared_ptr<const LValue> op, memory::Device device) const;
        virtual std::shared_ptr<const Runnable> sub_to(std::shared_ptr<const LValue> op, memory::Device device) const;
        virtual std::shared_ptr<const Runnable> mul_to(std::shared_ptr<const LValue> op, memory::Device device) const;
        virtual std::shared_ptr<const Runnable> div_to(std::shared_ptr<const LValue> op, memory::Device device) const;

        virtual std::shared_ptr<const Runnable> as_runnable(memory::Device device) const;
        virtual std::shared_ptr<const ArrayWrapper>    initialize_destination(memory::Device device) const;

        virtual std::shared_ptr<const Runnable> operator_to(OPERATOR_T operator_t, std::shared_ptr<const LValue> op, memory::Device device) const final;
    };

    struct LValue: virtual public ExpressionState {
        // assumes that op,destination_array() does not return NULL.
        virtual std::shared_ptr<const Runnable> assign_from(std::shared_ptr<const Runnable> op, memory::Device device) const = 0;
        virtual std::shared_ptr<const Runnable> add_from(std::shared_ptr<const Runnable> op, memory::Device device) const = 0;
        virtual std::shared_ptr<const Runnable> sub_from(std::shared_ptr<const Runnable> op, memory::Device device) const = 0;
        virtual std::shared_ptr<const Runnable> mul_from(std::shared_ptr<const Runnable> op, memory::Device device) const = 0;
        virtual std::shared_ptr<const Runnable> div_from(std::shared_ptr<const Runnable> op, memory::Device device) const = 0;

        virtual std::shared_ptr<const Runnable> operator_from(OPERATOR_T operator_t, std::shared_ptr<const Runnable> op, memory::Device device) const final;

    };

    struct LRValue: virtual public LValue, virtual public RValue {
        virtual std::shared_ptr<const Runnable> add_from(std::shared_ptr<const Runnable> op, memory::Device device) const;
        virtual std::shared_ptr<const Runnable> sub_from(std::shared_ptr<const Runnable> op, memory::Device device) const;
        virtual std::shared_ptr<const Runnable> mul_from(std::shared_ptr<const Runnable> op, memory::Device device) const;
        virtual std::shared_ptr<const Runnable> div_from(std::shared_ptr<const Runnable> op, memory::Device device) const;
    };


    struct Runnable : virtual public RValue {
        virtual void run() const = 0;
        // if and only if this operation is assignment to an array, return operation corresponding
        // to that array. Otherwise return NULL.
        virtual std::shared_ptr<const ExpressionState> destination_op() const = 0;

        virtual std::shared_ptr<const Runnable> assign_to(std::shared_ptr<const LValue> op, memory::Device device) const;
        virtual std::shared_ptr<const Runnable> add_to(std::shared_ptr<const LValue> op, memory::Device device) const;
        virtual std::shared_ptr<const Runnable> sub_to(std::shared_ptr<const LValue> op, memory::Device device) const;
        virtual std::shared_ptr<const Runnable> mul_to(std::shared_ptr<const LValue> op, memory::Device device) const;
        virtual std::shared_ptr<const Runnable> div_to(std::shared_ptr<const LValue> op, memory::Device device) const;

    };



    struct Expression {
        std::shared_ptr<const ExpressionState> state_;

        Expression(const Array& arr);

        Expression(const Assignable<Array>& arr);

        Expression(double scalar);
        Expression(float scalar);
        Expression(int scalar);

        Expression(std::shared_ptr<const ExpressionState> state);

        DType dtype() const;
        int ndim() const;
        memory::Device preferred_device() const;


        bool is_scalar() const;
        bool is_assignable() const;

        std::vector<int> bshape() const;
        std::vector<int> shape() const;
        std::string name() const;

        int number_of_elements() const;

        operator Assignable<Array>() const;
        operator Assignable<ArrayGather>() const;
        operator Assignable<ArraySubtensor>() const;
    };
} // namespace expression
#endif // DALI_ARRAY_OP_EXPRESSION_EXPRESSION_H
