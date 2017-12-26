#include "eye.h"
#include "dali/array/jit/jit_runner.h"
#include "dali/array/jit/jit_utils.h"
#include "dali/utils/make_message.h"

namespace op {
namespace jit {

struct Diag : public JITNode {
    static const hash_t optype_hash;
    Array diag_;
    Diag(Array diag, int rows, int cols) : JITNode(2, {rows, cols}, diag.dtype()), diag_(diag) {}
    virtual std::vector<Array> arguments() const {
        return {diag_,};
    }
    virtual memory::Device preferred_device() const {
        return diag_.preferred_device();
    }
    virtual std::string get_call_code_nd(const SymbolTable& symbol_table,
                                         const node_to_info_t& node_to_info,
                                         memory::DeviceT device_type) const {
        return utils::make_message(
            kernel_name(node_to_info), "(",
            as_jit_node(diag_)->get_call_code_nd(symbol_table, node_to_info, device_type), ", ",
            symbol_table.get_shape(this), ")");
    }

    virtual void compute_node_compilation_info(int desired_computation_rank,
                                               const std::vector<int>& desired_computation_shape,
                                               SymbolTable& symbol_table,
                                               node_to_info_t* node_to_info) const {
        (*node_to_info)[this].computation_rank = desired_computation_rank;
        (*node_to_info)[this].computation_shape = desired_computation_shape;
        symbol_table.declare_shape(this);
        op::jit::compute_node_compilation_info(diag_,
                                               1,
                                               {shape_[0]},
                                               symbol_table,
                                               node_to_info);
        utils::Hasher hasher;
        hasher.add(optype_hash)
              .add(desired_computation_rank)
              .add(node_to_info->at(diag_.expression().get()).hash);
        (*node_to_info)[this].hash = hasher.value();
    }

    virtual std::string kernel_name(const node_to_info_t& node_to_info) const {
        return utils::make_message("diag", node_to_info.at(this).computation_rank, "d");
    }

    virtual std::string prefix_code(const node_to_info_t& node_to_info,
                                    memory::DeviceT device_type) const {
        int ndim = node_to_info.at(this).computation_rank;
        return define_kernel(/*ndim=*/ndim,
                             /*has_shape=*/true,
                             /*arguments=*/{"diag",},
                             /*kernel=*/utils::make_message("query[", ndim - 1, "] == query[", ndim - 2, "] ? "
                                                            "diag_[query[", node_to_info.at(this).computation_rank - 1,"]] : 0"),
                             /*name=*/kernel_name(node_to_info));
    }

    virtual expression_ptr copy() const {
        return std::make_shared<Diag>(diag_, shape_[0], shape_[1]);
    }
};

const hash_t Diag::optype_hash = std::hash<std::string>()(typeid(Diag).name());

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
