#include "reshape.h"
#include "dali/array/jit/jit_runner.h"
#include "dali/array/jit/jit_utils.h"
#include "dali/array/shape.h"
#include "dali/utils/make_message.h"
#include "dali/utils/core_utils.h"

namespace op {
namespace jit {

struct ReshapeRestride : public JITNode {
    ReshapeRestride(Array array, const std::vector<int>& shape,
                    int offset, const std::vector<int>& strides) :
        JITNode(shape, array.dtype(), {array}, offset, strides) {
    }

    virtual int min_computation_rank() const override {
        return op::jit::min_computation_rank(arguments_[0]);
    }

    virtual memory::Device preferred_device() const override {
        return arguments_[0].preferred_device();
    }

    virtual std::string get_call_code_nd(const SymbolTable& symbol_table,
                                         memory::DeviceT device_type) const override {
        return generate_call_code_nd(this,
                                     kernel_name(),
                                     symbol_table, device_type,
                                     /*has_shape=*/true);
    }

    virtual expression_ptr _reshape(const std::vector<int>& new_shape,
                                    const Array* owner) const override {
        return std::make_shared<ReshapeRestride>(arguments_[0], new_shape, offset_, strides_);
    }

    virtual expression_ptr jit_right_fit_ndim(int ndim) const override {
        if (ndim == 1) {
            if (arguments_[0].ndim() <= 1) {
                return arguments_[0].expression();
            } else {
                return op::jit::jit_right_fit_ndim(arguments_[0], ndim);
            }
        }
        auto new_shape = collapsed_shape(shape_, ndim);
        if (new_shape == arguments_[0].shape()) {
            return arguments_[0].expression();
        }
        return std::make_shared<ReshapeRestride>(arguments_[0], new_shape, offset_, strides_);
    }

    virtual bool shape_required() const override {return true;}

    virtual std::string kernel_name() const {
        return utils::make_message("reshape", ndim(), "d");
    }

    virtual std::string prefix_code(memory::DeviceT device_type) const override {
        return define_kernel(/*ndim=*/ndim(),
                             /*has_shape=*/true,
                             /*arguments=*/{"array",},
                             /*kernel=*/"array_[index_to_dim(indices_to_offset(shape_, query), array_.shape())]",
                             /*name=*/kernel_name(),
                             /*is_assignable=*/false);
    }

    virtual expression_ptr copy() const override {
        return std::make_shared<ReshapeRestride>(arguments_[0], shape_, offset_, strides_);
    }
};

Array jit_view(const Array& array,
               const std::vector<int>& shape,
               int offset,
               const std::vector<int>& strides) {
    return Array(std::make_shared<ReshapeRestride>(array, shape, offset, strides));
}

struct BroadcastedReshape : public JITNode {
    std::vector<bool> broadcasted_;
    BroadcastedReshape(Array array, const std::vector<int>& shape,
                       const std::vector<bool>& broadcasted) :
        JITNode(shape, array.dtype(), {array}),
        broadcasted_(broadcasted) {}

    virtual memory::Device preferred_device() const override {
        return arguments_[0].preferred_device();
    }

    virtual std::string get_call_code_nd(const SymbolTable& symbol_table,
                                         memory::DeviceT device_type) const override {
        return generate_call_code_nd(this,
                                     kernel_name(),
                                     symbol_table, device_type,
                                     /*has_shape=*/true);
    }

    virtual void update_symbol_table(SymbolTable& symbol_table, node_to_info_t& node_to_info) const override {
        symbol_table.store_into_temporary(arguments_[0], node_to_info);
    }

    virtual void compilation_parameters(utils::Hasher& hasher) const override {
        for (auto val : broadcasted_) {
            hasher.add(int(val));
        }
    }

    virtual bool shape_required() const override {return true;}

    virtual std::string kernel_name() const {
        std::stringstream ss;
        ss << "broadcasted_reshape";
        for (int i = 0; i < ndim(); i++) {
            ss << (broadcasted_[i] ? "T" : "F");
        }
        return ss.str();
    }

    virtual std::string prefix_code(memory::DeviceT device_type) const override {
        std::vector<std::string> queries;
        for (int i = 0; i < ndim(); i++) {
            if (broadcasted_[i]) {
                queries.emplace_back("0");
            } else {
                queries.emplace_back(utils::make_message("query[", i, "]"));
            }
        }
        return define_kernel(/*ndim=*/ndim(),
                             /*has_shape=*/true,
                             /*arguments=*/{"array",},
                             /*kernel=*/utils::make_message("array_[{", utils::join(queries, ", "), "}]"),
                             /*name=*/kernel_name(),
                             /*is_assignable=*/false);
    }

    virtual expression_ptr copy() const override {
        return std::make_shared<BroadcastedReshape>(arguments_[0], shape_, broadcasted_);
    }
};

namespace {
    std::vector<int> expand_shape(std::vector<int> old_shape, int axis) {
        old_shape.insert(old_shape.begin() + axis, 1);
        return old_shape;
    }

    std::vector<int> squeeze_shape(std::vector<int> old_shape, int axis) {
        old_shape.erase(old_shape.begin() + axis, old_shape.begin() + axis + 1);
        return old_shape;
    }
}

struct ExpandDims : public JITNode {
    int axis_;
    ExpandDims(Array array, int axis) :
        JITNode(expand_shape(array.shape(), axis), array.dtype(), {array}),
        axis_(axis) {}

    virtual int min_computation_rank() const override {
        return arguments_[0].ndim() + 1;
    }

    virtual memory::Device preferred_device() const override {
        return arguments_[0].preferred_device();
    }

    virtual std::string get_call_code_nd(const SymbolTable& symbol_table,
                                         memory::DeviceT device_type) const override {
        return generate_call_code_nd(this,
                                     kernel_name(),
                                     symbol_table, device_type,
                                     /*has_shape=*/true);
    }

    virtual std::string kernel_name() const {
        return utils::make_message("expand_dims_", axis_, "_", ndim(), "d");
    }

    virtual void compilation_parameters(utils::Hasher& hasher) const override {
        hasher.add(axis_);
    }

    virtual bool shape_required() const override {return true;}

    virtual std::string prefix_code(memory::DeviceT device_type) const override {
        std::vector<std::string> queries;
        int query_index = 0;
        for (int i = 0; i < ndim(); i++) {
            if (i != axis_) {
                queries.emplace_back(utils::make_message("query[", query_index, "]"));
                query_index++;
            } else {
                queries.emplace_back("0");
            }
        }
        return define_kernel(/*ndim=*/ndim(),
                             /*has_shape=*/true,
                             /*arguments=*/{"array",},
                             /*kernel=*/utils::make_message("array_[{", utils::join(queries, ", "), "}]"),
                             /*name=*/kernel_name(),
                             /*is_assignable=*/false);
    }

    virtual expression_ptr _squeeze(int axis, const Array* owner) const override;

    virtual expression_ptr copy() const override {
        return std::make_shared<ExpandDims>(arguments_[0], axis_);
    }
};

struct Squeeze : public JITNode {
    int axis_;
    Squeeze(Array array, int axis) :
        JITNode(squeeze_shape(array.shape(), axis), array.dtype(), {array}),
        axis_(axis) {}

    virtual int min_computation_rank() const override {
        return std::max(1, arguments_[0].ndim() - 1);
    }

    virtual memory::Device preferred_device() const override {
        return arguments_[0].preferred_device();
    }

    virtual std::string get_call_code_nd(const SymbolTable& symbol_table,
                                         memory::DeviceT device_type) const override {
        return generate_call_code_nd(this,
                                     kernel_name(),
                                     symbol_table, device_type,
                                     /*has_shape=*/true);
    }

    virtual expression_ptr _expand_dims(int new_axis, const Array* owner) const override;

    virtual void compilation_parameters(utils::Hasher& hasher) const override {
        hasher.add(axis_);
    }

    virtual std::string kernel_name() const {
        return utils::make_message("squeeze_", axis_, "_", ndim(), "d");
    }

    virtual bool shape_required() const override {return true;}

    virtual std::string prefix_code(memory::DeviceT device_type) const override {
        std::vector<std::string> queries;
        for (int i = 0; i < ndim(); i++) {
            if (i != axis_) {
                queries.emplace_back(utils::make_message("query[", i, "]"));
            }
        }
        return define_kernel(/*ndim=*/ndim(),
                             /*has_shape=*/true,
                             /*arguments=*/{"array",},
                             /*kernel=*/utils::make_message("array_[{", utils::join(queries, ", "), "}]"),
                             /*name=*/kernel_name(),
                             /*is_assignable=*/false);
    }

    virtual expression_ptr copy() const override {
        return std::make_shared<Squeeze>(arguments_[0], axis_);
    }
};

expression_ptr Squeeze::_expand_dims(int new_axis, const Array* owner) const {
    if (new_axis == axis_) {
        return arguments_[0].expression();
    } else {
        return std::make_shared<ExpandDims>(copy(), new_axis);
    }
}

expression_ptr ExpandDims::_squeeze(int new_axis, const Array* owner) const {
    if (new_axis == axis_) {
        return arguments_[0].expression();
    } else {
        return std::make_shared<Squeeze>(copy(), new_axis);
    }
}

Array broadcasted_reshape(const Array& array,
                          const std::vector<int>& shape) {
    if (array.shape() == shape) {
        return array;
    }
    const auto& current_shape = array.shape();
    ASSERT2(current_shape.size() == shape.size(), utils::make_message(
        "new_shape for broadcasted_reshape must have the same dimensionality "
        "as the current shape (current_shape = ", current_shape,
        ", new_shape = ", shape, ")."));
    std::vector<bool> broadcasted(current_shape.size(), 0);
    for (int i = 0; i < current_shape.size(); i++) {
        if (current_shape[i] != shape[i]) {
            ASSERT2(current_shape[i] == 1, utils::make_message(
                "broadcasted dimension must have size 1, but on axis ", i,
                " got dimension with size ", current_shape[i], " (current_shape = ",
                current_shape, ", new_shape = ", shape, ")."));
            broadcasted[i] = true;
        } else {
            broadcasted[i] = false;
        }
    }
    return Array(std::make_shared<BroadcastedReshape>(array, shape, broadcasted));
}

Array expand_dims(const Array& array, int axis) {
    return Array(std::make_shared<ExpandDims>(array, axis));
}

Array squeeze(const Array& array, int axis) {
    return Array(std::make_shared<Squeeze>(array, axis));
}

}  // namespace jit
}  // namespace op
