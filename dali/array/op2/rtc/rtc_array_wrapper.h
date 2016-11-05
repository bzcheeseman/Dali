#ifndef DALI_ARRAY_OP_RTC_ARRAY_WRAPPER_H
#define DALI_ARRAY_OP_RTC_ARRAY_WRAPPER_H

#include <string>
#include <memory>
#include <vector>

#include "dali/array/op2/expression/array_wrapper.h"
#include "dali/array/op2/rtc/rtc_expression.h"

namespace expression {
namespace rtc {
    struct RtcArrayWrapper : virtual public RtcExpression {
        static const hash_t optype_hash;

        Array array_;

        virtual DType dtype() const;
        virtual std::vector<int> bshape() const;
        virtual int ndim() const;
        virtual std::string name() const;
        virtual bool is_assignable() const;
        virtual bool contiguous() const;
        virtual std::vector<int> shape() const;
        virtual int number_of_elements() const;


        RtcArrayWrapper(const Array& array);

        virtual void compute_node_compilation_info(int desired_computation_rank,
                                                   const std::vector<int>& desired_computation_shape,
                                                   std::vector<const RtcArrayWrapper*>* arrays,
                                                   std::vector<const ScalarWrapper*>* scalars,
                                                   node_to_info_t* node_to_info) const;

        virtual bool is_dim_collapsible_with_dim_minus_one(const int& dim) const;

        virtual std::shared_ptr<const RtcExpression> collapse_dim_with_dim_minus_one(const int& dim) const;

        virtual std::shared_ptr<const RtcExpression> transpose(const std::vector<int>& permutation) const;

        virtual std::string get_call_code_nd(const symbol_table_t& symbol_table, const node_to_info_t& node_to_info, memory::DeviceT device_type) const;
        virtual std::shared_ptr<const RtcExpression> as_jit() const;
        virtual std::shared_ptr<const ArrayWrapper> as_array() const;
    };
}  // namespace rtc
}  // namespace expression


#endif  // DALI_ARRAY_OP_RTC_ARRAY_WRAPPER_H
