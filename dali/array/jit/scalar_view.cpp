#include "scalar_view.h"

#include "dali/utils/assert2.h"
#include "dali/utils/print_utils.h"
#include "dali/utils/make_message.h"
#include "dali/array/shape.h"
#include "dali/array/array.h"
#include "dali/utils/hash_utils.h"


namespace op {
namespace jit {

const hash_t ScalarView::optype_hash = std::hash<std::string>()("ScalarView");

ScalarView::ScalarView(DType type) : JITNode(1, {}, type) {}

memory::Device ScalarView::preferred_device() const {
    return memory::default_preferred_device;
}


std::vector<Array> ScalarView::arguments() const {
    return {};
}

bool ScalarView::spans_entire_memory() const {
    return true;
}

std::string ScalarView::get_call_code_nd(const symbol_table_t& symbol_table,
										 const node_to_info_t& node_to_info,
										 memory::DeviceT device_type) const {
	return symbol_table.at(this);
}

void ScalarView::compute_node_compilation_info(int desired_computation_rank,
                                               const std::vector<int>& desired_computation_shape,
                                               std::vector<const BufferView*>* arrays,
                                               std::vector<const ScalarView*>* scalars,
                                               node_to_info_t* node_to_info) const {
    scalars->emplace_back(this);
    (*node_to_info)[this].computation_rank = desired_computation_rank;
    (*node_to_info)[this].hash = utils::Hasher().add(optype_hash)
    									        .add((int)dtype_)
    									        .add(desired_computation_rank).value();
}

struct ScalarInt32View : public ScalarView {
    int value_;
    ScalarInt32View(int value) : ScalarView(DTYPE_INT32), value_(value) {};
    ScalarInt32View(const ScalarInt32View& other) : ScalarInt32View(other.value_) {};
    virtual expression_ptr copy() const {return std::make_shared<ScalarInt32View>(*this);}
    const void* value_ptr() const {return &value_;}
};

struct ScalarFp32View : public ScalarView {
    float value_;
    ScalarFp32View(float value) : ScalarView(DTYPE_FLOAT), value_(value) {};
    ScalarFp32View(const ScalarFp32View& other) : ScalarFp32View(other.value_) {};
    virtual expression_ptr copy() const {return std::make_shared<ScalarFp32View>(*this);}
    const void* value_ptr() const {return &value_;}
};

struct ScalarFp64View : public ScalarView {
    double value_;
    ScalarFp64View(double value) : ScalarView(DTYPE_DOUBLE), value_(value) {};
    ScalarFp64View(const ScalarFp64View& other) : ScalarFp64View(other.value_) {};
    virtual expression_ptr copy() const {return std::make_shared<ScalarFp64View>(*this);}
    const void* value_ptr() const {return &value_;}
};

Array wrap_scalar(int value) {
	return Array(std::make_shared<ScalarInt32View>(value));
}
Array wrap_scalar(float value) {
	return Array(std::make_shared<ScalarFp32View>(value));
}
Array wrap_scalar(double value) {
	return Array(std::make_shared<ScalarFp32View>(value));
}

}
}
