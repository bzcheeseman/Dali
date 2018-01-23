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

const hash_t ScalarView::optype_hash = std::hash<std::string>()(typeid(ScalarView).name());

ScalarView::ScalarView(DType type) : JITNode(1, {}, type, {}) {}

memory::Device ScalarView::preferred_device() const {
    return memory::default_preferred_device;
}

std::string ScalarView::get_call_code_nd(const SymbolTable& symbol_table,
										 const node_to_info_t& node_to_info,
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

void ScalarView::compute_node_compilation_info(int desired_computation_rank,
                                               const std::vector<int>& desired_computation_shape,
                                               SymbolTable& symbol_table,
                                               node_to_info_t& node_to_info) const {
    symbol_table.declare_scalar(this);
    node_to_info[this].computation_rank = desired_computation_rank;
    node_to_info[this].hash = utils::Hasher().add(optype_hash)
    									     .add((int)dtype_)
    									     .add(desired_computation_rank).value();
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
    static const hash_t optype_hash;
    TileScalar(Array scalar, const std::vector<int>& shape) :
        JITNode(1, shape, scalar.dtype(), {scalar}) {
    }
    virtual bool spans_entire_memory() const override {
        return true;
    }

    virtual memory::Device preferred_device() const override {
        return arguments_[0].preferred_device();
    }

    virtual std::string name() const override {
        return utils::make_message("TileScalar[", shape_, "]");
    }

    virtual std::string get_call_code_nd(const SymbolTable& symbol_table,
                                         const node_to_info_t& node_to_info,
                                         memory::DeviceT device_type) const {
        return utils::make_message(
            kernel_name(node_to_info), "(",
            as_jit_node(arguments_[0])->get_call_code_nd(symbol_table, node_to_info, device_type),
            ", ", symbol_table.get_shape(this), ")");
    }

    virtual void compute_node_compilation_info(int desired_computation_rank,
                                               const std::vector<int>& desired_computation_shape,
                                               SymbolTable& symbol_table,
                                               node_to_info_t& node_to_info) const {
        node_to_info[this].computation_rank = desired_computation_rank;
        node_to_info[this].computation_shape = desired_computation_shape;
        op::jit::compute_node_compilation_info(arguments_[0],
                                               1,
                                               {1},
                                               symbol_table,
                                               node_to_info);
        utils::Hasher hasher;
        hasher.add(optype_hash)
              .add(desired_computation_rank)
              .add(node_to_info.at(arguments_[0].expression().get()).hash);
        node_to_info[this].hash = hasher.value();

    }

    virtual bool shape_required() const {return true;}

    std::string kernel_name(const node_to_info_t& node_to_info) const {
        return utils::make_message("tile_scalar", node_to_info.at(this).computation_rank, "d");
    }

    virtual std::string prefix_code(const node_to_info_t& node_to_info,
                                    memory::DeviceT device_type) const override {
        return define_kernel(/*ndim=*/node_to_info.at(this).computation_rank,
                             /*has_shape=*/true,
                             /*arguments=*/{"scalar",},
                             /*kernel=*/"scalar_[0]",
                             /*name=*/kernel_name(node_to_info),
                             /*is_assignable=*/false);
    }

    virtual expression_ptr _reshape(const std::vector<int>& new_shape, const Array* owner) const {
        return std::make_shared<TileScalar>(arguments_[0], new_shape);
    }

    virtual expression_ptr copy() const {
        return std::make_shared<TileScalar>(arguments_[0], shape_);
    }
};

const hash_t TileScalar::optype_hash = std::hash<std::string>()(typeid(TileScalar).name());

Array tile_scalar(Array node, const std::vector<int>& shape) {
    ASSERT2(node.is_scalar(), utils::make_message(
        "tile_scalar can only be called on a scalar array "
        "(got Array with shape = ", node.shape(), ")."));
    return Array(std::make_shared<TileScalar>(node, shape));
}

}

}
