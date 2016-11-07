#include "rtc_expression.h"

#include "dali/utils/make_message.h"
#include "dali/array/op2/expression/array_wrapper.h"
#include "dali/array/op2/rtc/scalar_wrapper.h"
#include "dali/array/op2/rtc/rtc_array_wrapper.h"
#include "dali/array/op2/rtc_utils.h"
#include "dali/array/function2/compiler.h"

#include "dali/array/op2/rtc/rtc_assign.h"

using utils::Hasher;

namespace expression {
namespace rtc {

    RtcExpression::RtcExpression(int min_computation_rank) :
            min_computation_rank_(min_computation_rank) {
    }


    std::string RtcExpression::prefix_code(const node_to_info_t& node_to_info, memory::DeviceT device_type) const {
        return "";
    }

    bool RtcExpression::is_dim_collapsible_with_dim_minus_one(const int& dim) const {
        return false;
    }

    std::shared_ptr<const RtcExpression> RtcExpression::collapse_dim_with_dim_minus_one(const int& dim) const {
        return jit_shared_from_this();
    }

    std::shared_ptr<const RtcExpression> RtcExpression::transpose(const std::vector<int>& permutation) const {
        ASSERT2(false, "Transpose not implemented for this Expression.");
        return jit_shared_from_this();
    }

    bool RtcExpression::is_assignable() const {
        return false;
    }

    std::shared_ptr<const RtcExpression> RtcExpression::jit_shared_from_this() const {
        return std::dynamic_pointer_cast<const RtcExpression>(shared_from_this());
    }

    std::shared_ptr<RtcExpression> RtcExpression::jit_shared_from_this() {
        return std::dynamic_pointer_cast<RtcExpression>(shared_from_this());
    }

    std::shared_ptr<const Runnable> RtcExpression::assign_to(std::shared_ptr<const LValue> op, memory::Device device) const {
        return std::make_shared<RtcAssignExpressionState>(
            op->as_jit(),
            OPERATOR_T_EQL,
            jit_shared_from_this(),
            device
        );
    }

    std::shared_ptr<const Runnable> RtcExpression::add_to(std::shared_ptr<const LValue> op, memory::Device device) const {
        return std::make_shared<RtcAssignExpressionState>(
            op->as_jit(),
            OPERATOR_T_ADD,
            jit_shared_from_this(),
            device
        );
    }
    std::shared_ptr<const Runnable> RtcExpression::sub_to(std::shared_ptr<const LValue> op, memory::Device device) const {
        return std::make_shared<RtcAssignExpressionState>(
            op->as_jit(),
            OPERATOR_T_SUB,
            jit_shared_from_this(),
            device
        );
    }
    std::shared_ptr<const Runnable> RtcExpression::mul_to(std::shared_ptr<const LValue> op, memory::Device device) const {
        return std::make_shared<RtcAssignExpressionState>(
            op->as_jit(),
            OPERATOR_T_MUL,
            jit_shared_from_this(),
            device
        );
    }
    std::shared_ptr<const Runnable> RtcExpression::div_to(std::shared_ptr<const LValue> op, memory::Device device) const {
        return std::make_shared<RtcAssignExpressionState>(
            op->as_jit(),
            OPERATOR_T_DIV,
            jit_shared_from_this(),
            device
        );
    }
}  // namespace rtc
}  // namespace expression
