#ifndef DALI_ARRAY_OP_RTC_SCALAR_WRAPPER_H
#define DALI_ARRAY_OP_RTC_SCALAR_WRAPPER_H

#include <memory>
#include <string>
#include <vector>

#include "dali/array/memory/device.h"
#include "dali/array/op2/rtc/rtc_expression.h"


namespace expression {
    struct ArrayWrapper;

namespace rtc {
    struct ScalarWrapper;

    struct ScalarWrapper : public RtcExpression {
        ScalarWrapper();
        static const hash_t optype_hash;

        virtual DType dtype() const = 0;
        virtual std::vector<int> bshape() const;

        virtual int ndim() const;
        virtual std::string name() const = 0;
        virtual const void* value_ptr() const = 0;

        virtual std::vector<int> shape() const;

        virtual int number_of_elements() const;


        virtual void compute_node_compilation_info(int desired_computation_rank,
                                                   const std::vector<int>& desired_computation_shape,
                                                   std::vector<const ArrayWrapper*>* arrays,
                                                   std::vector<const ScalarWrapper*>* scalars,
                                                   node_to_info_t* node_to_info) const;

        virtual bool is_dim_collapsible_with_dim_minus_one(const int& dim) const;

        virtual std::shared_ptr<const RtcExpression> transpose(const std::vector<int>& permutation) const;

        virtual std::string get_call_code_nd(const symbol_table_t& symbol_table, const node_to_info_t& node_to_info, memory::DeviceT device_type) const;
    };

    struct ScalarWrapperDouble : public ScalarWrapper {
        double value_;

        ScalarWrapperDouble(double value);

        std::string name() const;

        DType dtype() const;

        const void* value_ptr() const;
    };

    struct ScalarWrapperInteger : public ScalarWrapper {
        int value_;

        ScalarWrapperInteger(int value);

        std::string name() const;

        DType dtype() const;

        const void* value_ptr() const;
    };

    struct ScalarWrapperFloat : public ScalarWrapper {
        float value_;

        ScalarWrapperFloat(float value);

        virtual std::string name() const;

        virtual DType dtype() const;

        virtual const void* value_ptr() const;
    };

}  // namespace rtc
}  // namespace expression

#endif  // DALI_ARRAY_OP_RTC_SCALAR_WRAPPER_H
