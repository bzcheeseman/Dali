#include "gather_from_rows.h"
#include "dali/array/jit/jit.h"
#include "dali/array/jit/jit_utils.h"
#include "dali/array/op/elementwise_operation.h"
#include "dali/utils/make_message.h"

namespace {
    std::vector<int> op_shape(const std::vector<int>& indices_shape, const std::vector<int>& source_shape) {
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
            GatherFromRows(Array source, Array indices) :
                    JITNode(op_shape(indices.shape(), source.shape()), source.dtype(), {source, indices}) {}

            virtual int min_computation_rank() const override {
                return std::max(1, arguments_[0].ndim() - 1);
            }

            virtual std::string kernel_name() const {
                return utils::make_message("gather_from_rows_kernel_", ndim(), "d");
            }

            expression_ptr copy() const override {
                return std::make_shared<GatherFromRows>(arguments_[0], arguments_[1]);
            }

            virtual expression_ptr buffer_arg() const override {
                return copy();
            }

            bool is_assignable() const override {
                return arguments_[0].is_assignable();
            }

            virtual void assignment_access_modes(SymbolTable& symbol_table, OPERATOR_T operator_t) const override {
                if (arguments_[0].is_buffer()) {
                    symbol_table.notify_access_mode(arguments_[0], memory::AM_MUTABLE);
                } else {
                    static_as_jit_node(arguments_[0])->assignment_access_modes(symbol_table, operator_t);
                }
            }

            void prefix_code(memory::DeviceT device_type,
                             bool assignment_code,
                             insert_t insert) const {
                std::string kernel;
                if (ndim() == 1) {
                    kernel = "source_[{query[0], indices_[query[0]]}];";
                } else {
                    std::stringstream ss;
                    ss << "source_[{query[0]";
                    ASSERT2(arguments_[1].ndim() == 1, utils::make_message(
                        "computation_rank for indices should be 1 (got rank=", arguments_[1].ndim(), ")."));
                    ss << ", indices_[query[0]]";
                    for (int i = 1; i < arguments_[0].ndim() - 1; i++) {
                        ss << ", query[" << i << "]";
                    }
                    ss << "}]";
                    kernel = ss.str();
                }
                define_kernel(/*ndim=*/ndim(),
                              /*has_shape=*/true,
                              /*arguments=*/{"source", "indices"},
                              /*kernel=*/kernel,
                              /*name=*/kernel_name(),
                              /*is_assignable=*/assignment_code,
                              insert);
            }

            virtual void prefix_code(memory::DeviceT device_type, insert_t insert) const override {
                prefix_code(device_type, false, insert);
            }

            virtual void assignment_prefix_code(hash_t hash,
                                                       const std::vector<OPERATOR_T>& operators,
                                                       memory::DeviceT device_type,
                                                       const std::vector<int>& computation_ranks,
                                                       const std::vector<PARALLELISM_T>& parallelism_types,
                                                       const std::vector<bool>& assignment,
                                                       const std::vector<bool>& grid_keep_inner_dims,
                                                       insert_t insert) const override {
                JITNode::assignment_prefix_code(
                    hash, operators, device_type, computation_ranks,
                    parallelism_types, assignment, grid_keep_inner_dims, insert);
                prefix_code(device_type, true, insert);
            }

            virtual bool shape_required() const override {return true;}

            virtual std::string get_call_code_nd(
                    const SymbolTable& symbol_table,
                    memory::DeviceT device_type) const override {
                return generate_call_code_nd(this, kernel_name(),
                                             symbol_table, device_type,
                                             /*has_shape=*/true);
            }
        };
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
