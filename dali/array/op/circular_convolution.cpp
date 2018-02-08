#include "circular_convolution.h"
#include "dali/array/jit/jit_runner.h"
#include "dali/array/jit/jit_utils.h"
#include "dali/array/op/elementwise_operation.h"
#include "dali/utils/make_message.h"

namespace op {
    namespace jit {
        struct CircularConvolution : public JITNode {
            CircularConvolution(const Array& content, const Array& weights) :
                    JITNode(get_common_shape({content, weights}),
                            content.dtype(), {content, weights}) {}

            virtual int min_computation_rank() const override {
                return std::max(2,
                                std::max(
                                    op::jit::min_computation_rank(arguments_[0]),
                                    op::jit::min_computation_rank(arguments_[1])));
            }

            expression_ptr copy() const override {
                return std::make_shared<CircularConvolution>(arguments_[0], arguments_[1]);
            }

            void prefix_code(memory::DeviceT device_type, insert_t insert) const override {
                std::string kernel = (
                    "T res = static_cast<T>(0);\n"
                    "const int conv_size = shape_[ndim - 1];\n"
                    "const int& x = query[ndim - 1];\n"
                    "Shape<ndim> content_query = query;\n"
                    "Shape<ndim> weights_query = query;\n"
                    "int& shift_idx = weights_query[ndim - 1];\n"
                    "int& offset = content_query[ndim - 1];\n"
                    "#pragma clang loop vectorize(enable)\n"
                    "#pragma clang loop interleave(enable)\n"
                    "for (shift_idx = 0; shift_idx < conv_size; shift_idx++) {\n"
                    "    offset = x + shift_idx;\n"
                    "    if (offset >= conv_size) {\n"
                    "        offset -= conv_size;\n"
                    "    }\n"
                    "    res += content_[content_query] * weights_[weights_query];\n"
                    "}\n"
                    "return res;\n");
                define_kernel(/*ndim=*/std::max(1, ndim()),
                              /*has_shape=*/true,
                              /*arguments=*/{"content", "weights"},
                              /*kernel=*/kernel,
                              /*name=*/kernel_name(),
                              /*is_assignable=*/false,
                              insert);
            }

            bool is_axis_collapsible_with_axis_minus_one(int axis) const override {
                if (axis == ndim() - 1) {
                    return false;
                }
                return arguments_[0].is_axis_collapsible_with_axis_minus_one(axis) &&
                       arguments_[1].is_axis_collapsible_with_axis_minus_one(axis);
            }

            expression_ptr collapse_axis_with_axis_minus_one(int axis, const Array* owner) const override {
                return std::make_shared<CircularConvolution>(
                    arguments_[0].collapse_axis_with_axis_minus_one(axis),
                    arguments_[1].collapse_axis_with_axis_minus_one(axis));
            }

            virtual bool shape_required() const override {return true;}

            std::string kernel_name() const {
                return utils::make_message("circular_convolution_", std::max(1, ndim()), "d");
            }

            std::string get_call_code_nd(const SymbolTable& symbol_table,
                                         memory::DeviceT device_type) const override {
                return generate_call_code_nd(this,
                                             kernel_name(),
                                             symbol_table, device_type,
                                             /*has_shape=*/true);
            }
        };
    } // namespace jit
    Array circular_convolution(Array x, Array weights) {
        std::tie(x, weights) = ensure_arguments_compatible(x, weights, "circular_convolution", true);
        return Array(std::make_shared<jit::CircularConvolution>(x, weights));
    }
}  // namespace op
