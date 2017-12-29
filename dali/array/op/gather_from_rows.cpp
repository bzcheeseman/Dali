#include "gather_from_rows.h"
#include "dali/array/jit/jit_runner.h"
#include "dali/array/jit/jit_utils.h"
#include "dali/array/op/elementwise_operation.h"
#include "dali/utils/make_message.h"

namespace {
    std::vector<int> op_shape(const std::vector<int>& indices_shape, const std::vector<int>& source_shape) {
        std::vector<int> res(source_shape.begin() + 2, source_shape.end());
        if (indices_shape.size() > 0) {
            res.insert(res.begin(), indices_shape.begin(), indices_shape.end());
        } else {
            res.insert(res.begin(), 1);
        }
        return res;
    }
}

namespace op {
    namespace jit {
        struct GatherFromRows : public JITNode {
            static const hash_t optype_hash;
            Array source_, indices_;

            GatherFromRows(Array source, Array indices) :
                    JITNode(std::max(1, source.ndim() - 1), op_shape(indices.shape(), source.shape()), source.dtype()),
                    source_(source), indices_(indices) {}

            virtual std::string kernel_name(const node_to_info_t& node_to_info) const {
                return utils::make_message("gather_from_rows_kernel_", node_to_info.at(this).computation_rank, "d");
            }

            expression_ptr copy() const {
                return std::make_shared<GatherFromRows>(source_, indices_);
            }

            expression_ptr buffer_arg() const {
                return copy();
            }

            bool is_assignable() const {
                return source_.is_assignable();
            }

            std::string prefix_code(const node_to_info_t& node_to_info, memory::DeviceT device_type,
                                    bool assignment_code) const {
                int source_rank = node_to_info.at(source_.expression().get()).computation_rank;
                int indices_rank = node_to_info.at(indices_.expression().get()).computation_rank;
                int self_rank = node_to_info.at(this).computation_rank;
                std::string kernel;
                if (self_rank == 1) {
                    kernel = "source_[{query[0], indices_[query[0]]}];";
                } else {
                    std::stringstream ss;
                    ss << "source_[{query[0]";
                    ASSERT2(indices_rank == 1, utils::make_message(
                        "computation_rank for indices should be 1 (got rank=", indices_rank, ")."));
                    ss << ", indices_[query[0]]";
                    for (int i = 1; i < source_rank - 1; i++) {
                        ss << ", query[" << i << "]";
                    }
                    ss << "}]";
                    kernel = ss.str();
                }
                return define_kernel(/*ndim=*/node_to_info.at(this).computation_rank,
                                     /*has_shape=*/true,
                                     /*arguments=*/{"source", "indices"},
                                     /*kernel=*/kernel,
                                     /*name=*/kernel_name(node_to_info),
                                     /*is_assignable=*/assignment_code);
            }

            std::string prefix_code(const node_to_info_t& node_to_info, memory::DeviceT device_type) const {
                return prefix_code(node_to_info, device_type, false);
            }

            std::string assignment_prefix_code(OPERATOR_T operator_t,
                                               const node_to_info_t& node_to_info,
                                               memory::DeviceT device_type,
                                               int computation_rank) const {
                return (JITNode::assignment_prefix_code(operator_t, node_to_info, device_type, computation_rank) +
                        prefix_code(node_to_info, device_type, true));
            }

            std::vector<Array> arguments() const {
                return {source_, indices_};
            }

            void compute_node_compilation_info(
                    int desired_computation_rank,
                    const std::vector<int>& desired_computation_shape,
                    SymbolTable& symbol_table,
                    node_to_info_t* node_to_info) const {
                (*node_to_info)[this].computation_rank = desired_computation_rank;

                auto source_original_bshape = source_.shape();
                int source_ndim = source_original_bshape.size();
                // indices dim 1, dim2, etc... source dim 3, dim 4, etc...
                std::vector<int> source_shape(
                    desired_computation_shape.end() - (source_ndim - 2),
                    desired_computation_shape.end()
                );
                // add dim 1 & 2 of source back in (hidden from output by gather operation).
                if (source_original_bshape[0] == -1) {
                    source_original_bshape[0] = desired_computation_shape[0];
                }
                if (source_original_bshape[1] == -1) {
                    source_original_bshape[1] = desired_computation_shape[0];
                }
                source_shape.insert(source_shape.begin(), source_original_bshape.begin(), source_original_bshape.begin() + 2);
                std::vector<int> indices_shape(
                    desired_computation_shape.begin(),
                    desired_computation_shape.end() - (source_ndim - 2)
                );

                op::jit::compute_node_compilation_info(source_, source_ndim, source_shape, symbol_table, node_to_info);
                op::jit::compute_node_compilation_info(indices_, 1, indices_shape, symbol_table, node_to_info);
                symbol_table.declare_shape(this);
                (*node_to_info)[this].hash = utils::Hasher().add(optype_hash)
                                                            .add(desired_computation_rank)
                                                            .add(node_to_info->at(source_.expression().get()).hash)
                                                            .add(node_to_info->at(indices_.expression().get()).hash)
                                                            .value();
            }

            std::string get_call_code_nd(
                    const SymbolTable& symbol_table,
                    const node_to_info_t& node_to_info,
                    memory::DeviceT device_type) const {
                return utils::make_message(kernel_name(node_to_info), "(",
                                            op::jit::get_call_code_nd(source_, symbol_table, node_to_info, device_type),
                                            ", ",
                                            op::jit::get_call_code_nd(indices_, symbol_table, node_to_info, device_type),
                                            ", ", symbol_table.get_shape(this), ")");
            }
        };
        const hash_t GatherFromRows::optype_hash = std::hash<std::string>()(typeid(GatherFromRows).name());
    }  // namespace jit
    Array gather_from_rows(const Array& source, const Array& indices) {
        ASSERT2(source.ndim() > 1, utils::make_message(
            "gather must be called on source with ndim >= 2 (got ndim=", source.ndim(), ")."));
        ASSERT2(indices.dtype() == DTYPE_INT32, utils::make_message(
            "indices must be integers (got dtype=", indices.dtype(), ")."));
        ASSERT2(indices.ndim() <= 1, utils::make_message(
            "indices must have rank 1 or lower [Note: support for higher ranks coming soon] "
            "(got indices.ndim=", indices.ndim(), ")."));
        auto index_shape = indices.shape();
        auto source_shape = source.shape();
        if (index_shape.size() > 0) {
            ASSERT2(index_shape[0] <= source_shape[0] || index_shape[0] == 1 || source_shape[0] == 1,
                utils::make_message("dimension 1 of indices must be less than or equal "
                    "to first dimension of source (got indices.shape[0]=", index_shape[0],
                    ", source.shape[0]=", source_shape[0], ")")
            );
        }
        return Array(std::make_shared<jit::GatherFromRows>(source, indices));
    }
}  // namespace op
