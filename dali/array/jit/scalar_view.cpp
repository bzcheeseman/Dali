#include "scalar_view.h"

#include "dali/utils/assert2.h"
#include "dali/utils/print_utils.h"
#include "dali/utils/make_message.h"
#include "dali/array/shape.h"
#include "dali/array/array.h"
#include "dali/utils/hash_utils.h"
#include "dali/array/jit/jit_utils.h"


namespace op {
namespace jit {
ScalarView::ScalarView(DType type) : JITNode({}, type, {}) {}

memory::Device ScalarView::preferred_device() const {
    return memory::default_preferred_device;
}

std::string ScalarView::get_call_code_nd(const SymbolTable& symbol_table,
										 memory::DeviceT device_type) const {
	return symbol_table.get_name(this);
}

bool ScalarView::antialias() const {
    return false;
}

hash_t ScalarView::compute_node_data_hash(const node_to_info_t& node_to_info,
                                          const SymbolTable& symbol_table) const {
    return symbol_table.get_scalar_index(this);
}

void ScalarView::update_symbol_table(SymbolTable& symbol_table, node_to_info_t&) const {
    symbol_table.declare_scalar(this);
}

struct ScalarInt32View : public ScalarView {
    int value_;
    ScalarInt32View(int value) : ScalarView(DTYPE_INT32), value_(value) {};
    ScalarInt32View(const ScalarInt32View& other) : ScalarInt32View(other.value_) {};
    virtual expression_ptr copy() const {return std::make_shared<ScalarInt32View>(*this);}
    const void* value_ptr() const {return &value_;}
    virtual std::string name() const {
        return utils::make_message("int(", value_, ")");
    }
};

struct ScalarFp32View : public ScalarView {
    float value_;
    ScalarFp32View(float value) : ScalarView(DTYPE_FLOAT), value_(value) {};
    ScalarFp32View(const ScalarFp32View& other) : ScalarFp32View(other.value_) {};
    virtual expression_ptr copy() const {return std::make_shared<ScalarFp32View>(*this);}
    const void* value_ptr() const {return &value_;}
    virtual std::string name() const {
        return utils::make_message("fp32(", value_, ")");
    }
};

struct ScalarFp64View : public ScalarView {
    double value_;
    ScalarFp64View(double value) : ScalarView(DTYPE_DOUBLE), value_(value) {};
    ScalarFp64View(const ScalarFp64View& other) : ScalarFp64View(other.value_) {};
    virtual expression_ptr copy() const {return std::make_shared<ScalarFp64View>(*this);}
    const void* value_ptr() const {return &value_;}

    virtual std::string name() const {
        return utils::make_message("fp64(", value_, ")");
    }
};

Array wrap_scalar(int value) {
	return Array(std::make_shared<ScalarInt32View>(value));
}
Array wrap_scalar(float value) {
	return Array(std::make_shared<ScalarFp32View>(value));
}
Array wrap_scalar(double value) {
	return Array(std::make_shared<ScalarFp64View>(value));
}


struct TileScalar : public JITNode {
    TileScalar(Array scalar, const std::vector<int>& shape) :
        JITNode(shape, scalar.dtype(), {scalar}) {
    }
    virtual bool spans_entire_memory() const override {
        return true;
    }

    virtual int min_computation_rank() const override {
        return 1;
    }

    virtual memory::Device preferred_device() const override {
        return arguments_[0].preferred_device();
    }

    virtual std::string name() const override {
        return utils::make_message("TileScalar[", shape_, "]");
    }

    virtual std::string get_call_code_nd(const SymbolTable& symbol_table,
                                         memory::DeviceT device_type) const override {
        return utils::make_message(
            kernel_name(), "(",
            op::jit::get_call_code_nd(arguments_[0], symbol_table, device_type),
            ", ", symbol_table.get_shape(this), ")");
    }

    virtual bool shape_required() const override {return true;}

    std::string kernel_name() const {
        return utils::make_message("tile_scalar", ndim(), "d");
    }

    virtual void prefix_code(memory::DeviceT device_type, insert_t insert) const override {
        define_kernel(/*ndim=*/ndim(),
                      /*has_shape=*/true,
                      /*arguments=*/{"scalar",},
                      /*kernel=*/"scalar_[0]",
                      /*name=*/kernel_name(),
                      /*is_assignable=*/false,
                      insert);
    }

    virtual expression_ptr _reshape(const std::vector<int>& new_shape, const Array* owner) const override {
        return std::make_shared<TileScalar>(arguments_[0], new_shape);
    }

    virtual expression_ptr jit_right_fit_ndim(int ndim) const override {
        if (ndim == 1 && number_of_elements() == 1) {
            return arguments_[0].expression();
        }
        return std::make_shared<TileScalar>(arguments_[0], collapsed_shape(shape_, ndim));
    }

    virtual expression_ptr copy() const override {
        return std::make_shared<TileScalar>(arguments_[0], shape_);
    }
};

Array tile_scalar(Array node, const std::vector<int>& shape) {
    ASSERT2(node.is_scalar(), utils::make_message(
        "tile_scalar can only be called on a scalar array "
        "(got Array with shape = ", node.shape(), ")."));
    if (shape.size() == 0) return node;
    return Array(std::make_shared<TileScalar>(node, shape));
}

}

}
