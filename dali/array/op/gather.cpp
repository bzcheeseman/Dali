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

namespace op {
    namespace jit {
        struct Gather : public JITNode {
            static const hash_t optype_hash;
            Array source_, indices_;

            Gather(const Array& source, const Array& indices) :
                    JITNode(std::max(2, source.ndim() + min_computation_rank(indices) - 1),
                            gather_shape(source.shape(), indices.shape()), source.dtype()),
                    source_(source),
                    indices_(indices) {}

            std::string kernel_name(const node_to_info_t& node_to_info) const {
                return utils::make_message("gather_kernel", node_to_info.at(this).computation_rank, "d");
            }

            std::string prefix_code(const node_to_info_t& node_to_info, memory::DeviceT device_type) const {
                std::string kernel;
                bool is_2d = node_to_info.at(this).computation_rank == 2 &&
                             node_to_info.at(source_.expression().get()).computation_rank == 2;
                if (is_2d) {
                    kernel = "source_[{indices_[query[0]], query[1]}]";
                } else {
                    kernel = "Shape<C1::ndim> source_query = query.template axis_reduced_shape<C2::ndim, C1::ndim - 1, 1>();\n"
                             "source_query[0] = indices_[query.template axis_reduced_shape<0, C2::ndim>()];\n"
                             "return source_[source_query];\n";
                }
                return define_kernel(/*ndim=*/node_to_info.at(this).computation_rank,
                                     /*has_shape=*/true,
                                     /*arguments=*/{"source", "indices"},
                                     /*kernel=*/kernel,
                                     /*name=*/kernel_name(node_to_info));
            }

            std::vector<Array> arguments() const {
                return {source_, indices_};
            }

            bool is_axis_collapsible_with_axis_minus_one(int axis) const {
                int indices_ndim = indices_.ndim();
                if (axis < indices_ndim) {
                    return indices_.is_axis_collapsible_with_axis_minus_one(axis);
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
                    return source_.is_axis_collapsible_with_axis_minus_one(
                        axis - indices_ndim + 1
                    );
                }
                return false;
            }

            expression_ptr collapse_axis_with_axis_minus_one(int axis) const {
                int indices_ndim = indices_.ndim();
                if (axis < indices_ndim) {
                    return std::make_shared<Gather>(
                        source_,
                        indices_.collapse_axis_with_axis_minus_one(axis));
                } else {
                    return std::make_shared<Gather>(
                        source_.collapse_axis_with_axis_minus_one(axis - indices_ndim + 1),
                        indices_);
                }
            }

            void compute_node_compilation_info(
                    int desired_computation_rank,
                    const std::vector<int>& desired_computation_shape,
                    SymbolTable& symbol_table,
                    node_to_info_t* node_to_info) const {
                (*node_to_info)[this].computation_rank = desired_computation_rank;
                int source_ndim = source_.ndim();
                // indices dim 1, dim2, etc... source dim 2, dim 3, etc...
                std::vector<int> source_shape(
                    desired_computation_shape.end() - (source_ndim - 1),
                    desired_computation_shape.end()
                );
                // add dim 1 of source back in (hidden from output by gather operation).
                source_shape.insert(source_shape.begin(), source_.shape()[0]);
                std::vector<int> indices_shape(
                    desired_computation_shape.begin(),
                    desired_computation_shape.end() - (source_ndim - 1)
                );

                op::jit::compute_node_compilation_info(source_, source_ndim, source_shape, symbol_table, node_to_info);
                op::jit::compute_node_compilation_info(indices_, desired_computation_rank - (source_ndim - 1), indices_shape, symbol_table, node_to_info);
                symbol_table.declare_shape(this);
                bool is_2d = desired_computation_rank == 2 && source_ndim == 2;
                (*node_to_info)[this].hash = utils::Hasher().add(optype_hash)
                                                            .add(desired_computation_rank)
                                                            .add(is_2d)
                                                            .add(node_to_info->at(source_.expression().get()).hash)
                                                            .add(node_to_info->at(indices_.expression().get()).hash)
                                                            .value();
            }


            expression_ptr copy() const {
                return std::make_shared<Gather>(source_, indices_);
            }

            std::string get_call_code_nd(const SymbolTable& symbol_table,
                                         const node_to_info_t& node_to_info,
                                         memory::DeviceT device_type) const {
                return utils::make_message(kernel_name(node_to_info), "(",
                                            op::jit::get_call_code_nd(source_, symbol_table, node_to_info, device_type),
                                            ",",
                                            op::jit::get_call_code_nd(indices_, symbol_table, node_to_info, device_type),
                                            ",", symbol_table.get_shape(this), ")");
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
