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

            OneHot(Array on_value, Array off_value, Array indices, int depth) :
                    JITNode(one_hot_shape(indices.shape(), depth),
                            on_value.dtype(), {on_value, off_value, indices}) {
            }

            int min_computation_rank() const override {
                return op::jit::min_computation_rank(arguments_[2]) + 1;
            }

            std::string kernel_name(const node_to_info_t& node_to_info) const {
                return utils::make_message("one_hot", node_to_info.at(this).computation_rank, "d");
            }

            std::string prefix_code(const node_to_info_t& node_to_info,
                                    memory::DeviceT device_type) const override {
                return define_kernel(/*ndim=*/node_to_info.at(this).computation_rank,
                                     /*has_shape=*/true,
                                     /*arguments=*/{"on_value", "off_value", "indices"},
                                     /*kernel=*/"T is_on = indices_[query.template axis_reduced_shape<0, ndim-1>()] == query[ndim - 1];\n"
                                                "return on_value_[0] * is_on + (1.0 - is_on) * off_value_[0]",
                                     /*name=*/kernel_name(node_to_info),
                                     /*is_assignable=*/false);
            }

            expression_ptr copy() const override {
                return std::make_shared<OneHot>(arguments_[0], arguments_[1], arguments_[2], shape_.back());
            }

            virtual void compute_node_compilation_info(int desired_computation_rank,
                                                       const std::vector<int>& desired_computation_shape,
                                                       SymbolTable& symbol_table,
                                                       node_to_info_t& node_to_info) const override {
                node_to_info[this].computation_rank = desired_computation_rank;
                node_to_info[this].computation_shape = desired_computation_shape;
                op::jit::compute_node_compilation_info(arguments_[0], 1, {1}, symbol_table, node_to_info);
                op::jit::compute_node_compilation_info(arguments_[1], 1, {1}, symbol_table, node_to_info);
                op::jit::compute_node_compilation_info(arguments_[2], std::max(1, desired_computation_rank - 1), drop_last(desired_computation_shape), symbol_table, node_to_info);
                node_to_info[this].hash = utils::Hasher().add(optype_hash)
                                                         .add(desired_computation_rank)
                                                         .add(node_to_info.at(arguments_[0].expression().get()).hash)
                                                         .add(node_to_info.at(arguments_[1].expression().get()).hash)
                                                         .add(node_to_info.at(arguments_[2].expression().get()).hash)
                                                         .value();

            }

            virtual bool shape_required() const override {return true;}

            std::string get_call_code_nd(
                    const SymbolTable& symbol_table,
                    const node_to_info_t& node_to_info,
                    memory::DeviceT device_type) const override {
                return generate_call_code_nd(this, kernel_name(node_to_info),
                                             symbol_table, node_to_info, device_type,
                                             /*has_shape=*/true);
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
        std::tie(on_value, off_value) = ensure_arguments_compatible(on_value, off_value, "one_hot", false);
        return Array(std::make_shared<jit::OneHot>(on_value, off_value, indices, depth));
    }
}  // namespace op
