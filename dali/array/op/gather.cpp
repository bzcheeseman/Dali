#include "gather.h"
#include "dali/array/jit/jit_runner.h"
#include "dali/array/jit/jit_utils.h"
#include "dali/array/op/elementwise_operation.h"
#include "dali/utils/make_message.h"
namespace {
    std::vector<int> gather_shape(const std::vector<int>& source_shape, const std::vector<int>& indices_shape) {
        auto res = indices_shape;
        res.insert(res.end(), source_shape.begin() + 1, source_shape.end());
        return res;
    }
}

std::string operator_to_atomic(OPERATOR_T operator_t, std::string left, std::string right) {
    if (operator_t == OPERATOR_T_EQL) {
        return utils::make_message("atomicExch(&", left, ", ", right, ")");
    } else if (operator_t == OPERATOR_T_ADD) {
        return utils::make_message("atomicAdd(&", left, ", ", right, ")");
    } else if (operator_t == OPERATOR_T_SUB) {
        return utils::make_message("atomicAdd(&", left, ", -", right, ")");
    } else if (operator_t == OPERATOR_T_DIV) {
        return utils::make_message("atomicExch(&", left, ", ", left, " / ", right, ")");
    } else if (operator_t == OPERATOR_T_MUL) {
        return utils::make_message("atomicExch(&", left, ", ", left, " * ", right, ")");
    } else {
        ASSERT2(false, utils::make_message("no way to convert operator ", operator_to_name(operator_t), " into an atomic."));
    }
}

namespace op {
    namespace jit {
        struct Gather : public JITNode {
            static const hash_t optype_hash;
            Gather(const Array& source, const Array& indices) :
                    JITNode(std::max(2, source.ndim() + min_computation_rank(indices) - 1),
                            gather_shape(source.shape(), indices.shape()), source.dtype(),
                            {source, indices}) {}

            std::string kernel_name(const node_to_info_t& node_to_info) const {
                return utils::make_message("gather_kernel", node_to_info.at(this).computation_rank, "d");
            }

            std::string prefix_code(const node_to_info_t& node_to_info, memory::DeviceT device_type,
                                    bool assignment_code) const {
                std::string kernel;
                bool is_2d = node_to_info.at(this).computation_rank == 2 &&
                             node_to_info.at(arguments_[0].expression().get()).computation_rank == 2;
                if (is_2d) {
                    kernel = "source_[{indices_[query[0]], query[1]}]";
                } else {
                    kernel = "Shape<C1::ndim> source_query = query.template axis_reduced_shape<C2::ndim, C1::ndim - 1, 1>();\n"
                             "source_query[0] = indices_[query.template axis_reduced_shape<0, C2::ndim>()];\n"
                             "return source_[source_query];\n";
                }
                return define_kernel(
                    /*ndim=*/node_to_info.at(this).computation_rank,
                    /*has_shape=*/true,
                    /*arguments=*/{"source", "indices"},
                    /*kernel=*/kernel,
                    /*name=*/kernel_name(node_to_info),
                    /*is_assignable=*/assignment_code);
            }

            expression_ptr buffer_arg() const {
                return copy();
            }

            std::string prefix_code(const node_to_info_t& node_to_info, memory::DeviceT device_type) const {
                return prefix_code(node_to_info, device_type, false);
            }

            std::string assignment_prefix_code(OPERATOR_T operator_t,
                                               const node_to_info_t& node_to_info,
                                               memory::DeviceT device_type,
                                               int computation_rank) const {
                return (JITNode::assignment_prefix_code(operator_t, node_to_info, device_type, computation_rank) +
                        prefix_code(node_to_info, device_type, true));
            }

            std::string assignment_code_nd(OPERATOR_T operator_t, memory::DeviceT device_type,
                                           std::string dst, std::string src) const {
#ifdef DALI_USE_CUDA
                if (device_type == memory::DEVICE_T_GPU) {
                    return operator_to_atomic(operator_t, dst, src);
                }
#endif
                return utils::make_message(dst, " ", operator_to_name(operator_t), " ", src);
            }

            bool is_axis_collapsible_with_axis_minus_one(int axis) const {
                int indices_ndim = arguments_[1].ndim();
                if (axis < indices_ndim) {
                    return arguments_[1].is_axis_collapsible_with_axis_minus_one(axis);
                }
                if (axis == indices_ndim) {
                    // this is the dimensionality of the output's dimension just after
                    // the leading dimension. Collapsing this dimension means losing track
                    // of what is being gathered.
                    return false;
                }
                if (axis >= indices_ndim + 1) {
                    // because axis is being observed from the output, we must
                    // subtract all the index dimensions, and add back a dimension
                    // hidden by gather
                    return arguments_[0].is_axis_collapsible_with_axis_minus_one(
                        axis - indices_ndim + 1
                    );
                }
                return false;
            }

            bool is_assignable() const {
                return arguments_[0].is_assignable();
            }

            expression_ptr collapse_axis_with_axis_minus_one(int axis) const {
                int indices_ndim = arguments_[1].ndim();
                if (axis < indices_ndim) {
                    return std::make_shared<Gather>(
                        arguments_[0],
                        arguments_[1].collapse_axis_with_axis_minus_one(axis));
                } else {
                    return std::make_shared<Gather>(
                        arguments_[0].collapse_axis_with_axis_minus_one(axis - indices_ndim + 1),
                        arguments_[1]);
                }
            }

            void compute_node_compilation_info(
                    int desired_computation_rank,
                    const std::vector<int>& desired_computation_shape,
                    SymbolTable& symbol_table,
                    node_to_info_t* node_to_info) const {
                (*node_to_info)[this].computation_rank = desired_computation_rank;
                int source_ndim = arguments_[0].ndim();
                // indices dim 1, dim2, etc... source dim 2, dim 3, etc...
                std::vector<int> source_shape(
                    desired_computation_shape.end() - (source_ndim - 1),
                    desired_computation_shape.end()
                );
                // add dim 1 of source back in (hidden from output by gather operation).
                source_shape.insert(source_shape.begin(), arguments_[0].shape()[0]);
                std::vector<int> indices_shape(
                    desired_computation_shape.begin(),
                    desired_computation_shape.end() - (source_ndim - 1)
                );

                op::jit::compute_node_compilation_info(arguments_[0], source_ndim, source_shape, symbol_table, node_to_info);
                op::jit::compute_node_compilation_info(arguments_[1], desired_computation_rank - (source_ndim - 1), indices_shape, symbol_table, node_to_info);
                symbol_table.declare_shape(this);
                bool is_2d = desired_computation_rank == 2 && source_ndim == 2;
                (*node_to_info)[this].hash = utils::Hasher().add(optype_hash)
                                                            .add(desired_computation_rank)
                                                            .add(is_2d)
                                                            .add(node_to_info->at(arguments_[0].expression().get()).hash)
                                                            .add(node_to_info->at(arguments_[1].expression().get()).hash)
                                                            .value();
            }


            expression_ptr copy() const {
                return std::make_shared<Gather>(arguments_[0], arguments_[1]);
            }

            std::string get_call_code_nd(const SymbolTable& symbol_table,
                                         const node_to_info_t& node_to_info,
                                         memory::DeviceT device_type) const {
                return generate_call_code_nd(this,
                                         kernel_name(node_to_info),
                                         symbol_table, node_to_info, device_type,
                                         /*has_shape=*/true);
            }
        };
        const hash_t Gather::optype_hash = std::hash<std::string>()(typeid(Gather).name());
    }  // namespace jit
    Array gather(const Array& source, const Array& indices) {
        ASSERT2(source.ndim() > 0, utils::make_message(
            "gather must be called on source with ndim >= 1 (got ndim=", source.ndim(), ")."));
        ASSERT2(indices.dtype() == DTYPE_INT32, utils::make_message(
            "indices must be integers (got dtype=", indices.dtype(), ")."));
        return Array(std::make_shared<jit::Gather>(source, indices));
    }
}  // namespace op
