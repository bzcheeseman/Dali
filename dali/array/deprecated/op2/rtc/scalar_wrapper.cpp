#include "scalar_wrapper.h"

using utils::Hasher;

namespace expression {
namespace rtc {
    const hash_t ScalarWrapper::optype_hash = std::hash<std::string>()("ScalarWrapper");

    ScalarWrapper::ScalarWrapper() : RtcExpression(1) {}

    std::vector<int> ScalarWrapper::bshape() const {
        return {};
    }

    int ScalarWrapper::ndim() const {
        return 0;
    }

    std::string ScalarWrapper::name() const {
        return "double";
    }

    std::vector<int> ScalarWrapper::shape() const {
        return {};
    }

    int ScalarWrapper::number_of_elements() const {
        return 1;
    }

    void ScalarWrapper::compute_node_compilation_info(int desired_computation_rank,
                                                      const std::vector<int>& desired_computation_shape,
                                                      std::vector<const RtcArrayWrapper*>* arrays,
                                                      std::vector<const ScalarWrapper*>* scalars,
                                                      node_to_info_t* node_to_info) const {
        scalars->emplace_back(this);
        (*node_to_info)[this].computation_rank = desired_computation_rank;
        (*node_to_info)[this].hash = Hasher().add(optype_hash).add((int)dtype()).add(desired_computation_rank).value();
    }

    bool ScalarWrapper::is_dim_collapsible_with_dim_minus_one(const int& dim) const {
        return true;
    }

    std::shared_ptr<const RtcExpression> ScalarWrapper::transpose(const std::vector<int>& permutation) const {
        return jit_shared_from_this();
    }

    std::string ScalarWrapper::get_call_code_nd(const symbol_table_t& symbol_table, const node_to_info_t& node_to_info, memory::DeviceT device_type) const {
        return symbol_table.at(this);
    }




    ScalarWrapperDouble::ScalarWrapperDouble(double value) : ScalarWrapper(), value_(value) {}

    std::string ScalarWrapperDouble::name() const {
        return "double";
    }

    DType ScalarWrapperDouble::dtype() const {
        return DTYPE_DOUBLE;
    }

    const void* ScalarWrapperDouble::value_ptr() const {
        return (const void*)&value_;
    }




    ScalarWrapperInteger::ScalarWrapperInteger(int value) : ScalarWrapper(), value_(value) {}

    std::string ScalarWrapperInteger::name() const {
        return "int";
    }

    DType ScalarWrapperInteger::dtype() const {
        return DTYPE_INT32;
    }

    const void* ScalarWrapperInteger::value_ptr() const {
        return (const void*)&value_;
    }




    ScalarWrapperFloat::ScalarWrapperFloat(float value) : ScalarWrapper(), value_(value) {}

    std::string ScalarWrapperFloat::name() const {
        return "float";
    }

    DType ScalarWrapperFloat::dtype() const {
        return DTYPE_FLOAT;
    }

    const void* ScalarWrapperFloat::value_ptr() const {
        return (const void*)&value_;
    }
}  // namespace rtc
}  // namespace expression
