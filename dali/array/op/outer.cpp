#include "outer.h"
#include "dali/utils/make_message.h"
#include "dali/utils/assert2.h"
#include "dali/array/jit/jit.h"
#include "dali/array/jit/jit_utils.h"
#include "dali/array/op/elementwise_operation.h"
namespace op {
    namespace jit {
        struct Outer : public JITNode {
            Outer(Array left, Array right) : JITNode({left.shape()[0], right.shape()[0]}, left.dtype(), {left, right}) {}

            std::string kernel_name() const {
                return utils::make_message("outer", ndim(), "d");
            }

            void prefix_code(memory::DeviceT device_type, insert_t insert) const override {
                define_kernel(/*ndim=*/ndim(),
                              /*has_shape=*/true,
                              /*arguments=*/{"left", "right"},
                              /*kernel=*/"left_[query[ndim - 2]] * right_[query[ndim - 1]]",
                              /*name=*/kernel_name(),
                              /*is_assignable=*/false,
                              insert);
            }
            expression_ptr copy() const override {return std::make_shared<Outer>(arguments_[0], arguments_[1]);}

            virtual bool shape_required() const override {return true;}

            std::string get_call_code_nd(const SymbolTable& symbol_table,
                                         memory::DeviceT device_type) const override {
                return generate_call_code_nd(this, kernel_name(),
                                             symbol_table, device_type,
                                             /*has_shape=*/true);
            }
        };
    }  // namespace jit

    Array outer(Array a, Array b) {
        std::tie(a, b) = ensure_arguments_compatible(a.ravel(), b.ravel(), "outer",
            /*update_shape=*/false);
        return Array(std::make_shared<jit::Outer>(a, b));
    }
}  // namespace op
