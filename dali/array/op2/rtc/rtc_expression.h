#ifndef DALI_ARRAY_OP_RTC_EXPRESSION_H
#define DALI_ARRAY_OP_RTC_EXPRESSION_H

#include <unordered_map>
#include <unordered_set>
#include <string>
#include <vector>

#include "dali/array/op2/expression/expression.h"
#include "dali/utils/hash_utils.h"

namespace expression {
    namespace rtc {
        struct RtcArrayWrapper;
        struct ScalarWrapper;
    }
}

namespace expression {
namespace rtc {

    struct CompilationInfo {
        int  computation_rank;
        std::string name;
        std::vector<int> computation_shape;
        hash_t hash;
    };

    struct RtcExpression : virtual public RValue {
        typedef std::unordered_map<const ExpressionState*, std::string>     symbol_table_t;
        typedef std::unordered_map<const ExpressionState*, CompilationInfo> node_to_info_t;

        const int min_computation_rank_;

        RtcExpression() = delete;
        RtcExpression(int min_computation_rank);

        virtual void compute_node_compilation_info(int desired_computation_rank,
                                                   const std::vector<int>& desired_computation_shape,
                                                   std::vector<const RtcArrayWrapper*>* arrays,
                                                   std::vector<const ScalarWrapper*>* scalars,
                                                   node_to_info_t* node_to_info) const = 0;

        virtual std::string get_call_code_nd(const symbol_table_t& symbol_table, const node_to_info_t& node_to_info, memory::DeviceT device_type) const = 0;


        ///////////////////////////////////////////////////////////////////////////////
        //            REIMPLEMENT AS YOU SEE FIT                                     //
        ///////////////////////////////////////////////////////////////////////////////

        virtual std::string prefix_code(const node_to_info_t& node_to_info, memory::DeviceT device_type) const;


        // Returns true if striding is such that dim and (dim - 1) can be merged into
        // single dim. This function is allowed to returns false negatives, so if
        // your case is really complicated just return false (bear in mind that this
        // might sacrifice efficieny).
        virtual bool is_dim_collapsible_with_dim_minus_one(const int& dim) const;

        virtual std::shared_ptr<const RtcExpression> collapse_dim_with_dim_minus_one(const int& dim) const;

        virtual std::shared_ptr<const RtcExpression> transpose(const std::vector<int>& permutation) const;

        virtual bool is_assignable() const;

        ///////////////////////////////////////////////////////////////////////////////
        //            DO NOT REIMPLEMENT FUNCTIONS BELOW                             //
        ///////////////////////////////////////////////////////////////////////////////

        virtual std::shared_ptr<const RtcExpression> jit_shared_from_this() const final;
        virtual std::shared_ptr<RtcExpression> jit_shared_from_this() final;

        virtual std::shared_ptr<const Runnable> assign_to(std::shared_ptr<const LValue> op, memory::Device device) const;
        virtual std::shared_ptr<const Runnable> add_to(std::shared_ptr<const LValue> op, memory::Device device) const;
        virtual std::shared_ptr<const Runnable> sub_to(std::shared_ptr<const LValue> op, memory::Device device) const;
        virtual std::shared_ptr<const Runnable> mul_to(std::shared_ptr<const LValue> op, memory::Device device) const;
        virtual std::shared_ptr<const Runnable> div_to(std::shared_ptr<const LValue> op, memory::Device device) const;
    };
}  // namespace rtc
}  // namespace expression

#endif  // DALI_ARRAY_OP_RTC_EXPRESSION_H
