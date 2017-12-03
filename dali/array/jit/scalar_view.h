#ifndef DALI_ARRAY_EXPRESSION_SCALAR_VIEW_H
#define DALI_ARRAY_EXPRESSION_SCALAR_VIEW_H

#include <memory>
#include <vector>

#include "dali/array/jit/jit_runner.h"

namespace op {
namespace jit {

struct ScalarView : public JITNode {
	static const hash_t optype_hash;

    ScalarView(DType type);
    virtual std::vector<Array> arguments() const;
    virtual bool spans_entire_memory() const;
    virtual memory::Device preferred_device() const;
    virtual std::string get_call_code_nd(const symbol_table_t& symbol_table,
    									 const node_to_info_t& node_to_info,
    									 memory::DeviceT device_type) const;
    virtual void compute_node_compilation_info(int desired_computation_rank,
                                               const std::vector<int>& desired_computation_shape,
                                               std::vector<const BufferView*>* arrays,
                                               std::vector<const ScalarView*>* scalars,
                                               node_to_info_t* node_to_info) const;
    virtual expression_ptr copy() const = 0;
    virtual const void* value_ptr() const = 0;
};

Array wrap_scalar(int value);
Array wrap_scalar(float value);
Array wrap_scalar(double value);

}
}

#endif  // DALI_ARRAY_EXPRESSION_SCALAR_VIEW_H
