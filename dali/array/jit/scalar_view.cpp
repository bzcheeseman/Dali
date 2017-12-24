#include "scalar_view.h"

#include "dali/utils/assert2.h"
#include "dali/utils/print_utils.h"
#include "dali/utils/make_message.h"
#include "dali/array/shape.h"
#include "dali/array/array.h"
#include "dali/utils/hash_utils.h"


namespace op {
namespace jit {

const hash_t ScalarView::optype_hash = std::hash<std::string>()(typeid(ScalarView).name());

ScalarView::ScalarView(DType type) : JITNode(1, {}, type) {}

memory::Device ScalarView::preferred_device() const {
    return memory::default_preferred_device;
}


std::vector<Array> ScalarView::arguments() const {
    return {};
}

bool ScalarView::spans_entire_memory() const {
    return true;
}

std::string ScalarView::get_call_code_nd(const SymbolTable& symbol_table,
										 const node_to_info_t& node_to_info,
										 memory::DeviceT device_type) const {
	return symbol_table.get_name(this);
}

void ScalarView::compute_node_compilation_info(int desired_computation_rank,
                                               const std::vector<int>& desired_computation_shape,
                                               SymbolTable& symbol_table,
                                               node_to_info_t* node_to_info) const {
    symbol_table.declare_scalar(this);
    (*node_to_info)[this].computation_rank = desired_computation_rank;
    (*node_to_info)[this].hash = utils::Hasher().add(optype_hash)
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
    Array scalar_;
    TileScalar(Array scalar, const std::vector<int>& shape) :
        JITNode(1, shape, scalar.dtype()), scalar_(scalar) {
    }
    virtual std::vector<Array> arguments() const {
        return {scalar_};
    }
    virtual bool spans_entire_memory() const {
        return true;
    }
    virtual memory::Device preferred_device() const {
        return scalar_.preferred_device();
    }

    virtual std::string name() const {
        return utils::make_message("TileScalar[", shape_, "]");
    }

    virtual std::string get_call_code_nd(const SymbolTable& symbol_table,
                                         const node_to_info_t& node_to_info,
                                         memory::DeviceT device_type) const {
        return utils::make_message(
            kernel_name(node_to_info), "(",
            as_jit_node(scalar_)->get_call_code_nd(symbol_table, node_to_info, device_type),
            ", ", symbol_table.get_shape(this), ")");
    }

    virtual void compute_node_compilation_info(int desired_computation_rank,
                                               const std::vector<int>& desired_computation_shape,
                                               SymbolTable& symbol_table,
                                               node_to_info_t* node_to_info) const {
        (*node_to_info)[this].computation_rank = desired_computation_rank;
        (*node_to_info)[this].computation_shape = desired_computation_shape;
        symbol_table.declare_shape(this);
        op::jit::compute_node_compilation_info(scalar_,
                                               1,
                                               {1},
                                               symbol_table,
                                               node_to_info);
        utils::Hasher hasher;
        hasher.add(optype_hash)
              .add(desired_computation_rank)
              .add(node_to_info->at(scalar_.expression().get()).hash);
        (*node_to_info)[this].hash = hasher.value();
    }

    std::string kernel_name(const node_to_info_t& node_to_info) const {
        return utils::make_message("tile_scalar", node_to_info.at(this).computation_rank, "d");
    }

    virtual std::string prefix_code(const node_to_info_t& node_to_info,
                                    memory::DeviceT device_type) const {
        std::string name = utils::make_message(
            "TileScalarKernel", node_to_info.at(this).computation_rank, "D");

        return "template<typename C1>\n"
        "struct " + name + " {\n"
        "    const C1 scalar_;\n"
        "    static const int ndim = " + std::to_string(node_to_info.at(this).computation_rank) + ";\n"
        "    typedef typename C1::T T;\n"
        "    const Shape<ndim> shape_;\n"
        "    XINLINE const Shape<ndim>& shape() const {return shape_;}\n"
        "    XINLINE " + name + "(const C1& scalar, const Shape<ndim>& shape)"
        "       : scalar_(scalar), shape_(shape) {}\n"
        "    XINLINE T operator[](const Shape<ndim>& query) const {\n"
        "        return scalar_(0);\n"
        "    }\n"
        "    XINLINE T operator()(int i) const {return scalar_(0);}\n"
        "};\n"
        "template<typename C1>\n" +
        name + "<C1> " + kernel_name(node_to_info) +
        "(const C1& a, const Shape<" + std::to_string(node_to_info.at(this).computation_rank) + ">& b) {\n"
        "    return " + name + "<C1>(a, b);\n"
        "}\n";
    }

    virtual expression_ptr copy() const {
        return std::make_shared<TileScalar>(scalar_, shape_);
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
