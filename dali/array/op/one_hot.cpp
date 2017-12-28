#include "one_hot.h"
#include "dali/array/jit/jit_runner.h"
#include "dali/array/jit/jit_utils.h"
#include "dali/array/op/elementwise_operation.h"
#include "dali/utils/make_message.h"

namespace {
    std::vector<int> one_hot_shape(std::vector<int> base_shape, int new_dim) {
        base_shape.emplace_back(new_dim);
        return base_shape;
    }
    std::vector<int> drop_last(std::vector<int> base_shape) {
        ASSERT2(base_shape.size() > 1, utils::make_message(
            "Attempting to remove last dimension of indices in OneHot with ndim ",
            base_shape.size(), ", but indices must have at least ndim = 2"));
        base_shape.pop_back();
        return base_shape;
    }
}

namespace op {
    namespace jit {
        struct OneHot : public JITNode {
            static const hash_t optype_hash;
            Array indices_;
            Array on_value_;
            Array off_value_;

            OneHot(Array indices, int depth, Array on_value, Array off_value) :
                    JITNode(min_computation_rank(indices) + 1, one_hot_shape(indices.shape(), depth), on_value.dtype()),
                    indices_(indices), on_value_(on_value), off_value_(off_value) {
            }

            std::string kernel_name(const node_to_info_t& node_to_info) const {
                return utils::make_message("one_hot", node_to_info.at(this).computation_rank, "d");
            }

            std::string prefix_code(const node_to_info_t& node_to_info, memory::DeviceT device_type) const {
                return define_kernel(/*ndim=*/node_to_info.at(this).computation_rank,
                                     /*has_shape=*/true,
                                     /*arguments=*/{"on_value", "off_value", "indices"},
                                     /*kernel=*/"T is_on = indices_[query.template axis_reduced_shape<0, ndim-1>()] == query[ndim - 1];\n"
                                                "return on_value_[0] * is_on + (1.0 - is_on) * off_value_[0]",
                                     /*name=*/kernel_name(node_to_info));
            }

            std::vector<Array> arguments() const {return {on_value_, off_value_};}
            expression_ptr copy() const {return std::make_shared<OneHot>(indices_, on_value_, off_value_, shape_.back());}

            virtual void compute_node_compilation_info(int desired_computation_rank,
                                                       const std::vector<int>& desired_computation_shape,
                                                       SymbolTable& symbol_table,
                                                       node_to_info_t* node_to_info) const {
                (*node_to_info)[this].computation_rank = desired_computation_rank;
                (*node_to_info)[this].computation_shape = desired_computation_shape;
                symbol_table.declare_shape(this);
                op::jit::compute_node_compilation_info(indices_, desired_computation_rank - 1, drop_last(desired_computation_shape), symbol_table, node_to_info);
                op::jit::compute_node_compilation_info(on_value_, 1, {1}, symbol_table, node_to_info);
                op::jit::compute_node_compilation_info(off_value_, 1, {1}, symbol_table, node_to_info);
                (*node_to_info)[this].hash = utils::Hasher().add(optype_hash)
                                                            .add(desired_computation_rank)
                                                            .add(node_to_info->at(on_value_.expression().get()).hash)
                                                            .add(node_to_info->at(off_value_.expression().get()).hash)
                                                            .value();
            }

            std::string get_call_code_nd(
                    const SymbolTable& symbol_table,
                    const node_to_info_t& node_to_info,
                    memory::DeviceT device_type) const {
                return utils::make_message(kernel_name(node_to_info), "(",
                                           op::jit::get_call_code_nd(on_value_, symbol_table, node_to_info, device_type),
                                           ", ",
                                           op::jit::get_call_code_nd(off_value_, symbol_table, node_to_info, device_type),
                                           ", ",
                                           op::jit::get_call_code_nd(indices_, symbol_table, node_to_info, device_type),
                                           ", ",
                                           symbol_table.get_shape(this), ")");
            }
        };
        const hash_t OneHot::optype_hash = std::hash<std::string>()(typeid(OneHot).name());
    }  // namespace jit
    Array one_hot(Array indices, int depth, Array on_value, Array off_value) {
        ASSERT2(indices.dtype() == DTYPE_INT32, utils::make_message(
            "indices must be integers (got ", indices.dtype(), ")."));
        ASSERT2(on_value.is_scalar(), utils::make_message(
            "on_value must be a scalar (got on_value.ndim=", on_value.ndim(), ")."));
        ASSERT2(off_value.is_scalar(), utils::make_message(
            "off_value must be a scalar (got off_value.ndim=", off_value.ndim(), ")."));
        ASSERT2(depth > 0, utils::make_message(
            "depth must be strictly positive (got depth=", depth, ")."));
        std::tie(on_value, off_value) = ensure_arguments_compatible(on_value, off_value, "one_hot");
        return Array(std::make_shared<jit::OneHot>(indices, depth, on_value, off_value));
    }
}  // namespace op
