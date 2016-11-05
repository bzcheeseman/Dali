#include "rtc_array_wrapper.h"

using utils::Hasher;

namespace expression {
namespace rtc {
const hash_t RtcArrayWrapper::optype_hash = std::hash<std::string>()("ArrayWrapper");

DType RtcArrayWrapper::dtype() const {
    return array_.dtype();
}
std::vector<int> RtcArrayWrapper::bshape() const {
    return array_.bshape();
}
int RtcArrayWrapper::ndim() const {
    return array_.ndim();
}
std::string RtcArrayWrapper::name() const {
    return "Array";
}
bool RtcArrayWrapper::is_assignable() const {
    return true;
}
bool RtcArrayWrapper::contiguous() const {
    return array_.strides().empty();
}
std::vector<int> RtcArrayWrapper::shape() const {
    return array_.shape();
}
int RtcArrayWrapper::number_of_elements() const {
    return array_.number_of_elements();
}

std::shared_ptr<const RtcExpression> RtcArrayWrapper::collapse_dim_with_dim_minus_one(const int& dim) const {
    std::vector<int> newshape = array_.shape();
    newshape[dim - 1] = newshape[dim] * newshape[dim - 1];
    newshape.erase(newshape.begin() + dim);
    return std::make_shared<RtcArrayWrapper>(array_.copyless_reshape(newshape));
}

std::shared_ptr<const RtcExpression> RtcArrayWrapper::transpose(const std::vector<int>& permutation) const {
    return std::make_shared<RtcArrayWrapper>(array_.transpose(permutation));
}

std::string RtcArrayWrapper::get_call_code_nd(const symbol_table_t& symbol_table, const node_to_info_t& node_to_info, memory::DeviceT device_type) const {
    return symbol_table.at(this);
}

void RtcArrayWrapper::compute_node_compilation_info(int desired_computation_rank,
                                                    const std::vector<int>& desired_computation_shape,
                                                    std::vector<const RtcArrayWrapper*>* arrays,
                                                    std::vector<const ScalarWrapper*>* scalars,
                                                    node_to_info_t* node_to_info) const {
    arrays->emplace_back(this);
    (*node_to_info)[this].computation_rank  = desired_computation_rank;
    (*node_to_info)[this].computation_shape = desired_computation_shape;
    (*node_to_info)[this].hash = Hasher().add(optype_hash)
                                         .add(desired_computation_rank)
                                         .add(contiguous())
                                         .add(array_.dtype()).value();
}

bool RtcArrayWrapper::is_dim_collapsible_with_dim_minus_one(const int& dim) const {
    // TODO(jonathan): have fun and
    // make this check look at normalized strides
    // where possible (ensures that less code gets compiled)
    // once this is implemented, reshape needs to be updated
    // to leverage this property.
    return contiguous();
}

RtcArrayWrapper::RtcArrayWrapper(const Array& array) :
    RtcExpression(array.strides().empty() ? 1 : array.ndim()), array_(array) {
}

std::shared_ptr<const RtcExpression> RtcArrayWrapper::as_jit() const {
    return jit_shared_from_this();
}

}  // namespace rtc
}  // namespace expression

