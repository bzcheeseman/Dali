#include "one_hot.h"
#include "dali/array/jit/jit.h"
#include "dali/array/jit/jit_utils.h"
#include "dali/array/op/elementwise_operation.h"
#include "dali/utils/make_message.h"

namespace {
    std::vector<int> one_hot_shape(std::vector<int> base_shape, int new_dim) {
        base_shape.emplace_back(new_dim);
        return base_shape;
    }
}

namespace op {
    namespace jit {
        struct OneHot : public JITNode {
            OneHot(Array on_value, Array off_value, Array indices, int depth) :
                    JITNode(one_hot_shape(indices.shape(), depth),
                            on_value.dtype(), {on_value, off_value, indices}) {
            }

            int min_computation_rank() const override {
                return op::jit::min_computation_rank(arguments_[2]) + 1;
            }

            virtual expression_ptr jit_right_fit_ndim(int ndim) const override {
                return std::make_shared<OneHot>(
                    arguments_[0], arguments_[1],
                    op::jit::jit_right_fit_ndim(arguments_[2], ndim - 1),
                    shape_.back()
                );
            }

            std::string kernel_name() const {
                return utils::make_message("one_hot", std::max(1, ndim()), "d");
            }

            void prefix_code(memory::DeviceT device_type, insert_t insert) const override {
                define_kernel(/*ndim=*/std::max(1, ndim()),
                              /*has_shape=*/true,
                              /*arguments=*/{"on_value", "off_value", "indices"},
                              /*kernel=*/"T is_on = indices_[query.template axis_reduced_shape<0, ndim-1>()] == query[ndim - 1];\n"
                                         "return on_value_[0] * is_on + (1.0 - is_on) * off_value_[0]",
                              /*name=*/kernel_name(),
                              /*is_assignable=*/false,
                              insert);
            }

            expression_ptr copy() const override {
                return std::make_shared<OneHot>(arguments_[0], arguments_[1], arguments_[2], shape_.back());
            }

            virtual bool shape_required() const override {return true;}

            std::string get_call_code_nd(const SymbolTable& symbol_table,
                                         memory::DeviceT device_type) const override {
                return generate_call_code_nd(this, kernel_name(),
                                             symbol_table, device_type,
                                             /*has_shape=*/true);
            }
        };
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
