#ifndef DALI_ARRAY_EXPRESSION_SCALAR_VIEW_H
#define DALI_ARRAY_EXPRESSION_SCALAR_VIEW_H

#include <memory>
#include <vector>

#include "dali/utils/hash_utils.h"
#include "dali/array/jit/jit_runner.h"

namespace op {
namespace jit {

struct ScalarView : public JITNode {
    static const hash_t optype_hash;

    ScalarView(DType type);
    virtual memory::Device preferred_device() const override;
    virtual std::string get_call_code_nd(const SymbolTable& symbol_table,
                       const node_to_info_t& node_to_info,
                       memory::DeviceT device_type) const override;
    virtual void compute_node_compilation_info(int desired_computation_rank,
                                               const std::vector<int>& desired_computation_shape,
                                               SymbolTable& symbol_table,
                                               node_to_info_t& node_to_info) const override;
    virtual expression_ptr copy() const override = 0;
    virtual const void* value_ptr() const = 0;
    virtual bool antialias() const override;
    virtual hash_t compute_node_data_hash(const node_to_info_t& node_to_info, const SymbolTable& symbol_table) const override;
    virtual int min_computation_rank() const override;
};

Array wrap_scalar(int value);
Array wrap_scalar(float value);
Array wrap_scalar(double value);
Array tile_scalar(Array scalar, const std::vector<int>& shape);
}
}

#endif  // DALI_ARRAY_EXPRESSION_SCALAR_VIEW_H
