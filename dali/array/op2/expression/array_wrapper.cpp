#include "array_wrapper.h"

#include "dali/array/op2/rtc/rtc_array_wrapper.h"

namespace expression {
    ArrayWrapper::ArrayWrapper(Array array) :
            array_(array) {
    }

    DType ArrayWrapper::dtype() const {
        return array_.dtype();
    }

    std::string ArrayWrapper::name() const {
        return "Array";
    }

    std::vector<int> ArrayWrapper::bshape() const {
        return array_.bshape();
    }

    int ArrayWrapper::ndim() const {
        return array_.ndim();
    }

    bool ArrayWrapper::contiguous() const {
        return array_.strides().empty();
    }

    bool ArrayWrapper::is_assignable() const {
        return true;
    }

    std::vector<int> ArrayWrapper::shape() const {
        return array_.shape();
    }

    int ArrayWrapper::number_of_elements() const {
        return array_.number_of_elements();
    }

    std::shared_ptr<const Runnable> ArrayWrapper::as_runnable(memory::Device device) const {
        return std::dynamic_pointer_cast<const Runnable>(shared_from_this());
    }

    std::shared_ptr<const rtc::RtcExpression> ArrayWrapper::as_jit() const {
        return std::make_shared<rtc::RtcArrayWrapper>(array_);
    }

    std::shared_ptr<const ExpressionState> ArrayWrapper::destination_op() const {
        return shared_from_this();
    }

    void ArrayWrapper::run() const {
    }

    // void ArrayWrapper::compute_node_compilation_info(int desired_computation_rank,
    //                                                                 const std::vector<int>& desired_computation_shape,
    //                                                                 std::vector<const ArrayWrapper*>* arrays,
    //                                                                 std::vector<const ScalarWrapper*>* scalars,
    //                                                                 node_to_info_t* node_to_info) const {
    //     arrays->emplace_back(this);
    //     (*node_to_info)[this].computation_rank  = desired_computation_rank;
    //     (*node_to_info)[this].computation_shape = desired_computation_shape;
    //     (*node_to_info)[this].hash = Hasher().add(optype_hash)
    //                                          .add(desired_computation_rank)
    //                                          .add(contiguous())
    //                                          .add(array_.dtype()).value();
    // }

    // bool ArrayWrapper::is_dim_collapsible_with_dim_minus_one(const int& dim) const {
    //     // TODO(jonathan): have fun and
    //     // make this check look at normalized strides
    //     // where possible (ensures that less code gets compiled)
    //     // once this is implemented, reshape needs to be updated
    //     // to leverage this property.
    //     return contiguous();
    // }

    // std::shared_ptr<ExpressionState> ArrayWrapper::collapse_dim_with_dim_minus_one(const int& dim) const {
    //     std::vector<int> newshape = array_.shape();
    //     newshape[dim - 1] = newshape[dim] * newshape[dim - 1];
    //     newshape.erase(newshape.begin() + dim);
    //     return std::make_shared<ArrayWrapper>(array_.copyless_reshape(newshape));
    // }

    // std::shared_ptr<ExpressionState> ArrayWrapper::transpose(const std::vector<int>& permutation) const {
    //     return std::make_shared<ArrayWrapper>(array_.transpose(permutation));
    // }

    // std::string ArrayWrapper::get_call_code_nd(const symbol_table_t& symbol_table, const node_to_info_t& node_to_info, memory::DeviceT device_type) const {
    //     return symbol_table.at(this);
    // }


    std::shared_ptr<const Runnable> ArrayWrapper::assign_to(std::shared_ptr<const LValue> op, memory::Device device) const {
        ASSERT2(false, utils::make_message("Array's assignment to an lvalue (",
            op->full_operation_name(), ") is not yet implemented."));
    }

    std::shared_ptr<const Runnable> ArrayWrapper::add_to(std::shared_ptr<const LValue> op, memory::Device device) const {
        ASSERT2(false, utils::make_message("Array's addition-assignment to to an lvalue (",
            op->full_operation_name(), ") is not yet implemented."));
    }

    std::shared_ptr<const Runnable> ArrayWrapper::sub_to(std::shared_ptr<const LValue> op, memory::Device device) const {
        ASSERT2(false, utils::make_message("Array's substraction-assignment to to an lvalue (",
            op->full_operation_name(), ") is not yet implemented."));
    }

    std::shared_ptr<const Runnable> ArrayWrapper::mul_to(std::shared_ptr<const LValue> op, memory::Device device) const {
        ASSERT2(false, utils::make_message("Array's multiplication-assignment to to an lvalue (",
            op->full_operation_name(), ") is not yet implemented."));
    }

    std::shared_ptr<const Runnable> ArrayWrapper::div_to(std::shared_ptr<const LValue> op, memory::Device device) const {
        ASSERT2(false, utils::make_message("Array's division-assignment to an lvalue (",
            op->full_operation_name(), ") is not yet implemented."));
    }

    std::shared_ptr<const Runnable> ArrayWrapper::assign_from(std::shared_ptr<const Runnable> op, memory::Device device) const {
        ASSERT2(false, utils::make_message("Array's assignment by a runnable (",
            op->full_operation_name(), ") is not yet implemented."));
    }

    std::shared_ptr<const Runnable> ArrayWrapper::add_from(std::shared_ptr<const Runnable> op, memory::Device device) const {
        ASSERT2(false, utils::make_message("Array's addition by a runnable (",
            op->full_operation_name(), ") is not yet implemented."));
    }

    std::shared_ptr<const Runnable> ArrayWrapper::sub_from(std::shared_ptr<const Runnable> op, memory::Device device) const {
        ASSERT2(false, utils::make_message("Array's substraction by a runnable (",
            op->full_operation_name(), ") is not yet implemented."));
    }

    std::shared_ptr<const Runnable> ArrayWrapper::mul_from(std::shared_ptr<const Runnable> op, memory::Device device) const {
        ASSERT2(false, utils::make_message("Array's multiplication by a runnable (",
            op->full_operation_name(), ") is not yet implemented."));
    }

    std::shared_ptr<const Runnable> ArrayWrapper::div_from(std::shared_ptr<const Runnable> op, memory::Device device) const {
        ASSERT2(false, utils::make_message("Array's division by a runnable (",
            op->full_operation_name(), ") is not yet implemented."));
    }
}  // namespace expression
