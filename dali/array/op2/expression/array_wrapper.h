#ifndef DALI_ARRAY_OP_EXPRESSION_ARRAY_WRAPPER_H
#define DALI_ARRAY_OP_EXPRESSION_ARRAY_WRAPPER_H

#include <memory>
#include <string>
#include <vector>

#include "dali/array/memory/device.h"
#include "dali/array/op2/expression/expression.h"
#include "dali/utils/hash_utils.h"

struct Array;

namespace expression {

    struct ArrayWrapper : virtual public LRValue, virtual public Runnable {
        static const hash_t optype_hash;

        Array array_;

        ArrayWrapper(Array array);

        virtual DType dtype() const;

        virtual std::vector<int> bshape() const;

        virtual int ndim() const;
        virtual std::string name() const;
        virtual bool is_assignable() const;

        virtual bool contiguous() const;

        virtual std::vector<int> shape() const;

        virtual int number_of_elements() const;

        // virtual void compute_node_compilation_info(int desired_computation_rank,
        //                                            const std::vector<int>& desired_computation_shape,
        //                                            std::vector<const ArrayWrapper*>* arrays,
        //                                            std::vector<const ScalarWrapper*>* scalars,
        //                                            node_to_info_t* node_to_info) const;

        // virtual bool is_dim_collapsible_with_dim_minus_one(const int& dim) const;

        // virtual std::shared_ptr<ExpressionState> collapse_dim_with_dim_minus_one(const int& dim) const;

        // virtual std::shared_ptr<ExpressionState> transpose(const std::vector<int>& permutation) const;

        // virtual std::string get_call_code_nd(const symbol_table_t& symbol_table, const node_to_info_t& node_to_info, memory::DeviceT device_type) const;

        virtual std::shared_ptr<const Runnable> assign_to(std::shared_ptr<const LValue> op, memory::Device device) const;
        virtual std::shared_ptr<const Runnable> add_to(std::shared_ptr<const LValue> op, memory::Device device) const;
        virtual std::shared_ptr<const Runnable> sub_to(std::shared_ptr<const LValue> op, memory::Device device) const;
        virtual std::shared_ptr<const Runnable> mul_to(std::shared_ptr<const LValue> op, memory::Device device) const;
        virtual std::shared_ptr<const Runnable> div_to(std::shared_ptr<const LValue> op, memory::Device device) const;

        virtual std::shared_ptr<const Runnable> assign_from(std::shared_ptr<const Runnable> op, memory::Device device) const;
        virtual std::shared_ptr<const Runnable> add_from(std::shared_ptr<const Runnable> op, memory::Device device) const;
        virtual std::shared_ptr<const Runnable> sub_from(std::shared_ptr<const Runnable> op, memory::Device device) const;
        virtual std::shared_ptr<const Runnable> mul_from(std::shared_ptr<const Runnable> op, memory::Device device) const;
        virtual std::shared_ptr<const Runnable> div_from(std::shared_ptr<const Runnable> op, memory::Device device) const;

        virtual std::shared_ptr<const Runnable> as_runnable(memory::Device device) const;
        virtual void run() const;
        virtual std::shared_ptr<const ExpressionState> destination_op() const;
    };
}

#endif  // DALI_ARRAY_OP_EXPRESSION_ARRAY_WRAPPER_H

