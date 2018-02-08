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
            Gather(const Array& source, const Array& indices) :
                    JITNode(gather_shape(source.shape(), indices.shape()), source.dtype(),
                            {source, indices}) {}

            int min_computation_rank() const override {
                return std::max(2, arguments_[0].ndim() + op::jit::min_computation_rank(arguments_[1]) - 1);
            }

            std::string kernel_name() const {
                return utils::make_message("gather_kernel", ndim(), "d");
            }

            void prefix_code(memory::DeviceT device_type,
                             bool assignment_code,
                             insert_t insert) const {
                std::string kernel;
                bool is_2d = ndim() == 2 && arguments_[0].ndim() == 2;
                if (is_2d) {
                    kernel = "source_[{indices_[query[0]], query[1]}]";
                } else {
                    kernel = "Shape<C1::ndim> source_query = query.template axis_reduced_shape<C2::ndim, C1::ndim - 1, 1>();\n"
                             "source_query[0] = indices_[query.template axis_reduced_shape<0, C2::ndim>()];\n"
                             "return source_[source_query];\n";
                }
                define_kernel(
                    /*ndim=*/ndim(),
                    /*has_shape=*/true,
                    /*arguments=*/{"source", "indices"},
                    /*kernel=*/kernel,
                    /*name=*/kernel_name(),
                    /*is_assignable=*/assignment_code,
                    insert);
            }

            virtual expression_ptr buffer_arg() const override {
                return copy();
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

            virtual std::string assignment_code_nd(OPERATOR_T operator_t, memory::DeviceT device_type,
                                                   std::string dst, std::string src) const override  {
#ifdef DALI_USE_CUDA
                if (device_type == memory::DEVICE_T_GPU) {
                    return operator_to_atomic(operator_t, dst, src);
                }
#endif
                return utils::make_message(dst, " ", operator_to_name(operator_t), " ", src);
            }

            virtual bool is_axis_collapsible_with_axis_minus_one(int axis) const override {
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

            virtual bool is_assignable() const override {
                return arguments_[0].is_assignable();
            }

            virtual void assignment_access_modes(SymbolTable& symbol_table, OPERATOR_T operator_t) const override {
                if (arguments_[0].is_buffer()) {
                    symbol_table.notify_access_mode(arguments_[0], memory::AM_MUTABLE);
                } else {
                    static_as_jit_node(arguments_[0])->assignment_access_modes(symbol_table, operator_t);
                }
            }

            virtual expression_ptr collapse_axis_with_axis_minus_one(int axis, const Array* owner) const override {
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

            virtual void compilation_parameters(utils::Hasher& hasher) const override {
                hasher.add(ndim() == 2 && arguments_[0].ndim() == 2);
            }

            virtual bool shape_required() const override {return true;}

            expression_ptr copy() const override {
                return std::make_shared<Gather>(arguments_[0], arguments_[1]);
            }

            virtual std::string get_call_code_nd(const SymbolTable& symbol_table,
                                                 memory::DeviceT device_type) const override {
                return generate_call_code_nd(this,
                                         kernel_name(),
                                         symbol_table, device_type,
                                         /*has_shape=*/true);
            }
        };
    }  // namespace jit
    Array gather(const Array& source, const Array& indices) {
        ASSERT2(source.ndim() > 0, utils::make_message(
            "gather must be called on source with ndim >= 1 (got ndim=", source.ndim(), ")."));
        ASSERT2(indices.dtype() == DTYPE_INT32, utils::make_message(
            "indices must be integers (got dtype=", indices.dtype(), ")."));
        return Array(std::make_shared<jit::Gather>(source, indices));
    }
}  // namespace op
