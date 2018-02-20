#include "eye.h"
#include "dali/array/jit/jit.h"
#include "dali/array/jit/jit_utils.h"
#include "dali/utils/make_message.h"

namespace op {
namespace jit {
struct Diag : public JITNode {
    Diag(Array diag, int rows, int cols) : JITNode({rows, cols}, diag.dtype(), {diag}) {}

    virtual memory::Device preferred_device() const override {
        return arguments_[0].preferred_device();
    }

    virtual std::string get_call_code_nd(const SymbolTable& symbol_table,
                                         memory::DeviceT device_type) const override {
        return utils::make_message(
            kernel_name(), "(",
            as_jit_node(arguments_[0])->get_call_code_nd(symbol_table, device_type), ", ",
            symbol_table.get_shape(this), ")");
    }

    virtual bool shape_required() const override {return true;}

    virtual std::string kernel_name() const {
        return utils::make_message("diag", ndim(), "d");
    }

    virtual void prefix_code(memory::DeviceT device_type, insert_t insert) const override {
        define_kernel(/*ndim=*/ndim(),
                      /*has_shape=*/true,
                      /*arguments=*/{"diag",},
                      /*kernel=*/utils::make_message("query[", ndim() - 1, "] == query[", ndim() - 2, "] ? "
                                                     "diag_[query[", ndim() - 1,"]] : 0"),
                      /*name=*/kernel_name(),
                      /*is_assignable=*/false,
                      insert);
    }

    virtual expression_ptr copy() const override {
        return std::make_shared<Diag>(arguments_[0], shape_[0], shape_[1]);
    }
};
}  // namespace jit

Array eye(int size) {
    return eye(size, size);
}
Array eye(int rows, int cols) {
    ASSERT2(rows > 0 & cols > 0, utils::make_message(
        "eye's rows and cols must > 0 "
        "(got rows = ", rows, ", cols = ", cols, ")."));
    return Array(std::make_shared<jit::Diag>(float(1.0), rows, cols));
}

Array diag(Array array) {
    return diag(array, array.shape()[0], array.shape()[0]);
}

Array diag(Array array, int rows) {
    return diag(array, rows, rows);
}

Array diag(Array array, int rows, int cols) {
    ASSERT2(rows > 0 & cols > 0, utils::make_message(
        "diag's rows and cols must > 0 "
        "(got rows = ", rows, ", cols = ", cols, ")."));
    if (array.is_vector()) {
        ASSERT2(array.shape()[0] >= std::min(rows, cols), utils::make_message(
          "diag's argument is smaller than the maximal diagonal item "
          "(array.shape = ", array.shape(), ", rows = ", rows, ", cols = ", cols, ")."));
    }
    ASSERT2(array.is_vector() | array.is_scalar(), utils::make_message(
        "diag's argument must be a scalar or a vector "
        "(got array.shape = ", array.shape(), ")."));
    return Array(std::make_shared<jit::Diag>(array, rows, cols));
}
}  // namespace op
