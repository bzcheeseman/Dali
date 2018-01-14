#include "jit_runner.h"

#include <unordered_set>
#include <algorithm>

#include "dali/utils/make_message.h"
#include "dali/utils/assert2.h"
#include "dali/array/expression/optimization.h"
#include "dali/array/expression/assignment.h"
#include "dali/array/expression/computation.h"
#include "dali/array/expression/control_flow.h"
#include "dali/array/jit/jit_utils.h"
#include "dali/array/jit/reshape.h"
#include "dali/utils/compiler.h"
#include "dali/utils/scope.h"
#include "dali/array/op/binary.h"
#include "dali/array/op/unary.h"
#include "dali/array/op/elementwise_operation.h"
#include "dali/array/jit/scalar_view.h"

namespace op {
namespace jit {

bool should_always_recompile_is_cached = false;
bool should_always_recompile_cache     = false;

bool should_always_recompile() {
    if (!should_always_recompile_is_cached) {
        auto env_var_ptr = std::getenv("DALI_JIT_ALWAYS_RECOMPILE");
        std::string dali_jit_always_recompile;
        if (env_var_ptr == NULL) {
            dali_jit_always_recompile = "false";
        } else {
            dali_jit_always_recompile = env_var_ptr;
        }

        // lower
        for (int i = 0; i < dali_jit_always_recompile.size(); ++i) {
            if ('A' <= dali_jit_always_recompile[i] && dali_jit_always_recompile[i] <= 'Z') {
                dali_jit_always_recompile[i] += 'a' - 'A';
            }
        }
        should_always_recompile_cache = (dali_jit_always_recompile == "true");
        should_always_recompile_is_cached = true;
    }
    return should_always_recompile_cache;
}

// CONVENIENCE METHODS //

bool buffer_requires_strides(const Expression* buffer, const std::vector<int>& shape) {
    if (buffer->shape_.size() == 0) {
        return false;
    }
    if (buffer->strides_.size() > 0) {
        return true;
    }
    for (int i = 0; i < std::min(shape.size(), buffer->shape_.size()); i++) {
        if (buffer->shape_[i] != shape[i]) {
            return true;
        }
    }
    return false;
}

std::shared_ptr<JITNode> as_jit_node(Array array) {
    auto casted = std::dynamic_pointer_cast<JITNode>(array.expression());
    ASSERT2(casted != nullptr, utils::make_message("Attempting to cast a non-jit expression (",
        array.expression_name(), ") into a JITNode."));
    return casted;
}

JITNode* static_as_jit_node(const Array& array) {
    return static_cast<JITNode*>(array.expression().get());
}

hash_t node_hash(const node_to_info_t& node_to_info, const Array& array) {
    return node_to_info.at(array.expression().get()).hash;
}

void SymbolTable::declare_array(const BufferView* ptr) {
    arrays_.emplace_back(ptr);
}

void SymbolTable::declare_scalar(const ScalarView* ptr) {
    scalars_.emplace_back(ptr);
}

void SymbolTable::declare_shape(const Expression* ptr) {
    shapes_.emplace_back(ptr);
}

std::string SymbolTable::get_name(const Expression* ptr) const {
    auto name_pos = declaration_table_.find(ptr);
    ASSERT2(name_pos != declaration_table_.end(), utils::make_message(
        "No name was declared for expression ", ptr->full_name(),
        ".\nDon't forget to call `symbol_table.declare_array(this)/declare_scalar(this)`"
        "inside compute_node_compilation_info."));
    return name_pos->second;
}

std::string SymbolTable::get_shape(const Expression* ptr) const {
    auto name_pos = shape_declaration_table_.find(ptr);
    ASSERT2(name_pos != shape_declaration_table_.end(), utils::make_message(
        "No shape was declared for expression ", ptr->full_name(),
        ".\nDon't forget to call `symbol_table.declare_shape(this)` inside compute_node_compilation_info."));
    return name_pos->second;
}

std::string SymbolTable::variable_declarations(const node_to_info_t& node_to_info) const {
    std::stringstream result;
    for (int i = 0; i < arrays_.size(); ++i) {
        auto name = utils::make_message("array_", i, "_view");
        declaration_table_[(const Expression*)arrays_[i]] = name;
        if (buffer_requires_strides(arrays_[i], node_to_info.at((const Expression*)arrays_[i]).computation_shape)) {
            result << build_array_definition(
                dtype_to_cpp_name(arrays_[i]->dtype_),
                name,
                false,
                node_to_info.at((const Expression*)arrays_[i]).computation_rank,
                utils::make_message("array_data[", i, "], offsets[", i, "], sizes[", i, "], strides[", i, "]")
            );
        } else {
            result << build_array_definition(
                dtype_to_cpp_name(arrays_[i]->dtype_),
                name,
                true,
                node_to_info.at((const Expression*)arrays_[i]).computation_rank,
                utils::make_message("array_data[", i, "], offsets[", i, "], sizes[", i, "]")
            );
        }
    }

    for (int i = 0; i < scalars_.size(); ++i) {
        auto name = utils::make_message("scalar_", i, "_view");
        declaration_table_[(const Expression*)scalars_[i]] = name;
        result << build_scalar_definition(
            dtype_to_cpp_name(scalars_[i]->dtype_),
            name,
            node_to_info.at(scalars_[i]).computation_rank,
            utils::make_message("scalar_arguments[", i, "]")
        );
    }

    for (int i = 0; i < shapes_.size(); ++i) {
        auto name = utils::make_message("shape_", i);
        shape_declaration_table_[(const Expression*)shapes_[i]] = name;
        result << build_shape_definition(
            name,
            node_to_info.at(shapes_[i]).computation_rank,
            utils::make_message("shapes[", i, "]")
        );
    }

    return result.str();
}

std::vector<Array> SymbolTable::collect_buffers(const node_to_info_t& node_to_info) const {
    std::vector<Array> arrays;
    std::transform(arrays_.begin(),
                   arrays_.end(),
                   std::back_inserter(arrays),
                   [&node_to_info](const BufferView* op) {
                       const auto& rank  = node_to_info.at(op).computation_rank;
                       const auto& shape = node_to_info.at(op).computation_shape;
                       if (op->ndim() == 0) {
                          return op->broadcast_scalar_to_ndim(rank, nullptr);
                       }
                       if (rank == op->ndim()) {
                           return op->broadcast_to_shape(shape, nullptr);
                       } else if (rank == 1) {
                           return op->broadcast_to_shape(shape, nullptr)->ravel(nullptr);
                       } else {
                           return op->broadcast_to_shape(shape, nullptr)->right_fit_ndim(rank, nullptr);
                       }
                   });
    return arrays;
}

std::vector<const void*> SymbolTable::collect_scalars(const node_to_info_t& node_to_info) const {
    std::vector<const void*> scalars;
    std::transform(scalars_.begin(),
                   scalars_.end(),
                   std::back_inserter(scalars),
                   [&](const ScalarView* op) {
                        return op->value_ptr();
                   });
    return scalars;
}

std::vector<std::vector<int>> SymbolTable::collect_shapes(const node_to_info_t& node_to_info) const {
    std::vector<std::vector<int>> shapes;
    std::transform(shapes_.begin(),
                   shapes_.end(),
                   std::back_inserter(shapes),
                   [&](const Expression* op) {
                        const auto& rank  = node_to_info.at(op).computation_rank;
                        const auto& shape = node_to_info.at(op).computation_shape;
                        // because ranks may have changed from definition to compilation
                        // all shapes are reformated for runtime
                        if (rank == shape.size()) {
                            // no change
                            return shape;
                        } else if (rank == 1) {
                            // flatten
                            return op->ravel(nullptr)->shape_;
                        } else {
                            // flatten rightmost portion
                            return op->right_fit_ndim(rank, nullptr)->shape_;
                        }
                   });
    return shapes;
}

JITNode::JITNode(int min_computation_rank,
                 const std::vector<int>& shape,
                 DType dtype,
                 const std::vector<Array>& arguments,
                 int offset,
                 const std::vector<int>& strides) : Expression(shape, dtype, arguments, offset, strides),
                                                    min_computation_rank_(min_computation_rank) {
    ASSERT2(min_computation_rank > 0, utils::make_message(
        "JITNode computation rank must be greater than 0 (got "
        "min_computation_rank = ", min_computation_rank, ")."));
}

JITNode::JITNode(const JITNode& other) : Expression(other),
                                         min_computation_rank_(other.min_computation_rank_) {};

std::string JITNode::prefix_code(const node_to_info_t& node_to_info, memory::DeviceT device_type) const {
    return "";
}

bool JITNode::is_axis_collapsible_with_axis_minus_one(int axis) const {
    return false;
}

expression_ptr JITNode::_reshape(const std::vector<int>& new_shape, const Array*) const {
    // customize the way reshapes get run on JIT-nodes:
    // run them in place without a copy.
    return op::jit::jit_view(copy(), new_shape, 0, {}).expression();
}

memory::Device JITNode::preferred_device() const {
    memory::Device best_device = memory::Device::device_of_doom();
    // TODO(jonathan): ensure this logic actually picks the right device
    // to run based on the inputs, not just agreement
    for (auto& arg : arguments()) {
        auto new_pref_device = arg.preferred_device();
        if (best_device.is_error()) {
            best_device = new_pref_device;
        } else {
            if (new_pref_device == best_device) {
                best_device = new_pref_device;
            } else {
                best_device = memory::default_preferred_device;
                break;
            }
        }
    }
    return best_device;
}

bool JITNode::supports_operator(OPERATOR_T operator_t) const {
    return true;
}

std::string JITNode::assignment_code_nd(OPERATOR_T operator_t, memory::DeviceT device_type,
                                        std::string dst, std::string src) const {
    return utils::make_message(dst, " ", operator_to_name(operator_t), " ", src);
}

std::string JITNode::assignment_prefix_code(OPERATOR_T operator_t,
                                            const node_to_info_t& node_to_info,
                                            memory::DeviceT device_type,
                                            int computation_rank) const {
#ifdef DALI_USE_CUDA
    if (device_type == memory::DEVICE_T_GPU) {
        if (computation_rank == 1) {
            return utils::make_message(
                "template<typename Destination, typename Source>\n"
                "void __global__\n",
                "assign_kernel(Destination dst, Source src, int num_el) {\n"
                "    int idx = blockDim.x * blockIdx.x + threadIdx.x;\n"
                "    int stride = blockDim.x * gridDim.x;\n"
                "    for (int i = idx; i < num_el; i += stride) {\n"
                "        ", assignment_code_nd(operator_t, device_type, "dst[i]", "src[i]"), ";\n"
                "    }\n"
                "}\n"
            );
        } else {
            return utils::make_message(
                "template<typename Destination, typename Source>\n"
                "void __global__\n",
                "assign_kernel(Destination dst, Source src, int num_el, Shape<", computation_rank, "> shape) {\n"
                "    int idx = blockDim.x * blockIdx.x + threadIdx.x;\n"
                "    int stride = blockDim.x * gridDim.x;\n"
                "    for (int i = idx; i < num_el; i += stride) {\n"
                "        auto nd_idx = index_to_dim(idx, shape);\n"
                "        ", assignment_code_nd(operator_t, device_type, "dst[nd_idx]", "src[nd_idx]"), ";\n"
                "    }\n"
                "}\n"
            );
        }
    }
#endif
    return "";
}

expression_ptr only_buffers(std::shared_ptr<Expression> expr) {
    auto self_copy = expr->copy();
    for (auto& arg : self_copy->arguments()) {
        auto buffer_arg = arg.buffer_arg();
        if (!buffer_arg.is_stateless()) {
            arg.set_expression(buffer_arg.expression());
        }
    }
    return self_copy;
}

struct JITRunner : public JITNode {
    static const hash_t optype_hash;
    Array dest_, root_;
    const OPERATOR_T operator_t_;

    virtual expression_ptr copy() const;
    JITRunner(Array root, const std::vector<Array>& leaves, OPERATOR_T operator_t, Array dest);
    virtual memory::Device preferred_device() const;
    virtual void compute_node_compilation_info(int desired_computation_rank,
                                               const std::vector<int>& desired_computation_shape,
                                               SymbolTable& symbol_table,
                                               node_to_info_t* node_to_info) const;
    virtual bool is_axis_collapsible_with_axis_minus_one(const int& axis) const;
    virtual std::string get_call_code_nd(const SymbolTable& symbol_table,
                                         const node_to_info_t& node_to_info,
                                         memory::DeviceT device_type) const;

    std::string get_code_template(memory::Device device,
                                  const SymbolTable& symbol_table,
                                  const node_to_info_t& node_to_info) const;

    std::function<void(void**, const int*, const int**, const int**, const void**, const int**)> compile(
        memory::Device,
        const SymbolTable& symbol_table,
        const node_to_info_t& node_to_info) const;
    virtual std::string name() const;
};

const hash_t JITRunner::optype_hash = std::hash<std::string>()(typeid(JITRunner).name());

bool is_jit_node(const Array& array) {
    auto node = std::dynamic_pointer_cast<JITNode>(array.expression());
    return node != nullptr;
}

bool is_jit_runner(const Array& array) {
    auto node = std::dynamic_pointer_cast<JITRunner>(array.expression());
    return node != nullptr;
}

bool is_jit_assignment(const Array& node) {
    return (node.is_assignment() &&
            is_jit_node(static_as_assignment(node)->right()) &&
            !is_jit_runner(static_as_assignment(node)->right()));
}

std::shared_ptr<JITRunner> as_jit_runner(const Array& array) {
    return std::dynamic_pointer_cast<JITRunner>(array.expression());
}

// JIT RUNNER //
JITRunner::JITRunner(Array root, const std::vector<Array>& leaves, OPERATOR_T operator_t, Array dest) :
        JITNode(as_jit_node(root)->min_computation_rank_, dest.shape(), root.dtype(), leaves),
        root_(root), operator_t_(operator_t), dest_(dest) {
    if (is_jit_runner(root)) {
        throw std::runtime_error("JITRunner should not contain a JITRunner.");
    }
}

expression_ptr JITRunner::copy() const {
    return std::make_shared<JITRunner>(*this);
}

memory::Device JITRunner::preferred_device() const {
    return root_.preferred_device();
}

bool JITRunner::is_axis_collapsible_with_axis_minus_one(const int& axis) const {
    return static_as_jit_node(root_)->is_axis_collapsible_with_axis_minus_one(axis);
}

static hash_t BUFFER_HASH = std::hash<std::string>()(typeid(BufferView).name());


void buffer_compute_node_compilation_info(const Array& array, const Array& buffer_array,
                                          int desired_computation_rank,
                                          const std::vector<int>& desired_computation_shape,
                                          SymbolTable& symbol_table,
                                          node_to_info_t* node_to_info) {
    const BufferView* buffer_ptr = static_cast<const BufferView*>(buffer_array.expression().get());
    symbol_table.declare_array(buffer_ptr);
    (*node_to_info)[buffer_ptr].computation_rank  = desired_computation_rank;
    (*node_to_info)[buffer_ptr].computation_shape = desired_computation_shape;
    (*node_to_info)[buffer_ptr].hash = utils::Hasher().add(BUFFER_HASH)
                                                      .add(desired_computation_rank)
                                                      .add(buffer_requires_strides(buffer_ptr, desired_computation_shape))
                                                      .add(array.dtype()).value();
}


void JITRunner::compute_node_compilation_info(int desired_computation_rank,
                                              const std::vector<int>& desired_computation_shape,
                                              SymbolTable& symbol_table,
                                              node_to_info_t* node_to_info) const {
    (*node_to_info)[this].computation_rank = desired_computation_rank;

    op::jit::compute_node_compilation_info(dest_,
                                           desired_computation_rank,
                                           desired_computation_shape,
                                           symbol_table,
                                           node_to_info);
    op::jit::compute_node_compilation_info(root_,
                                           desired_computation_rank,
                                           desired_computation_shape,
                                           symbol_table,
                                           node_to_info);
    utils::Hasher hasher;
    hasher.add(optype_hash)
          .add(desired_computation_rank)
          .add(node_to_info->at(dest_.expression().get()).hash)
          .add(node_to_info->at(root_.expression().get()).hash)
          .add(int(operator_t_));
    (*node_to_info)[this].hash = hasher.value();
}

std::string JITRunner::get_call_code_nd(const SymbolTable& symbol_table,
                                        const node_to_info_t& node_to_info,
                                        memory::DeviceT device_type) const {
    return "";
}

std::string JITNode::assignment_code(const Array& dest,
                                     const Array& root,
                                     OPERATOR_T operator_t,
                                     const SymbolTable& symbol_table,
                                     const node_to_info_t& node_to_info,
                                     memory::DeviceT device_type,
                                     int computation_rank) const {
    auto jit_root = static_as_jit_node(root);
    std::string root_call = jit_root->get_call_code_nd(symbol_table, node_to_info, device_type);
    std::string dest_call_code = op::jit::get_call_code_nd(dest, symbol_table, node_to_info, device_type);
    if (device_type == memory::DEVICE_T_CPU) {
        std::string indexing_nd = computation_rank == 1 ? "[i]" : "[" + generate_accessor_string(computation_rank) + "]";
        std::string assign_line = assignment_code_nd(
            operator_t, device_type, utils::make_message(dest_call_code, indexing_nd),
            utils::make_message(root_call, indexing_nd));
        if (computation_rank == 1) {
            return utils::make_message(
                "    int num_el = ", dest_call_code, ".shape().numel();\n",
                "    #pragma clang loop vectorize(enable)\n",
                "    #pragma clang loop interleave(enable)\n",
                "    for (int i = 0; i < num_el; ++i) {\n",
                "        ", assign_line, ";\n"
                "    }\n"
            );
        } else {
            return construct_for_loop(
                computation_rank, assign_line, dest_call_code, 4);
        }
    }
#ifdef DALI_USE_CUDA
    else if (device_type == memory::DEVICE_T_GPU) {
        if (computation_rank == 1) {
            return utils::make_message(
                "    int num_el = ", dest_call_code, ".shape().numel();\n"
                "    const int NT = 128;\n"
                "    // const int MAX_BLOCKS = 40960;\n"
                "    int grid_size = div_ceil(num_el, NT);\n"
                "    // assert(grid_size <= MAX_BLOCKS);\n"
                "    assign_kernel<<<grid_size, NT, 0, NULL>>>(\n"
                "        ", dest_call_code, ",\n"
                "        ", root_call, ",\n"
                "        num_el\n"
                "    );\n"
            );
        } else {
            return utils::make_message(
                "    auto shape = ", dest_call_code, ".shape();\n"
                "    int num_el = shape.numel();\n"
                "    const int NT = 128;\n"
                "    // const int MAX_BLOCKS = 40960;\n"
                "    int grid_size = div_ceil(num_el, NT);\n"
                "    // assert(grid_size <= MAX_BLOCKS);\n"
                "    assign_kernel<<<grid_size, NT, 0, NULL>>>(\n"
                "        ", dest_call_code, ",\n"
                "        ", root_call, ",\n"
                "        num_el, shape\n"
                "    );\n"
            );
        }
    }
#endif
    else {
        ASSERT2(false, "unknown device type.");
    }
}

std::string JITRunner::get_code_template(memory::Device device,
                                         const SymbolTable& symbol_table,
                                         const node_to_info_t& node_to_info) const {
    std::unordered_set<hash_t> prefix_code_visited;
    std::stringstream result;
    auto insert_once = [&](const std::string& pc) {
        auto pc_hash = utils::get_hash(pc);
        if (prefix_code_visited.find(pc_hash) == prefix_code_visited.end()) {
            result << pc;
            prefix_code_visited.insert(pc_hash);
        }
    };
    int computation_rank = node_to_info.at(this).computation_rank;
    if (!dest_.is_buffer()) {
        insert_once(as_jit_node(dest_)->assignment_prefix_code(operator_t_, node_to_info, device.type(), computation_rank));
    } else {
        insert_once(assignment_prefix_code(operator_t_, node_to_info, device.type(), computation_rank));
    }
    auto add_prefix_code = [&](const Array& arr) {
        if (is_jit_node(arr)) {
            insert_once(
                static_as_jit_node(arr)->prefix_code(node_to_info, device.type()));
        }
    };
    dest_.expression()->for_all_suboperations(add_prefix_code);
    add_prefix_code(root_);
    root_.expression()->for_all_suboperations(add_prefix_code);
    result << "void run(void** array_data, const int* offsets, "
              "const int** sizes, const int** strides, "
              "const void** scalar_arguments, const int** shapes) {\n";
    // DECLARE SYMBOLS
    result << symbol_table.variable_declarations(node_to_info) << "\n";
    if (!dest_.is_buffer()) {
        result << as_jit_node(dest_)->assignment_code(dest_, root_, operator_t_, symbol_table, node_to_info, device.type(),
                                                      computation_rank);
    } else {
        result << assignment_code(dest_, root_, operator_t_, symbol_table, node_to_info, device.type(),
                                  computation_rank);
    }
    result << "}\n";
    return result.str();
}

std::function<void(void**, const int*, const int**, const int**, const void**, const int**)> JITRunner::compile(
            memory::Device device,
            const SymbolTable& symbol_table,
            const node_to_info_t& node_to_info) const {
    DALI_SCOPE("get_function");
    // compute a quasi-unique hash for the fused operation
    hash_t hash = utils::Hasher().add((int)device.type())
                                 .add(node_to_info.at(this).hash)
                                 .value();
    // check if the operation needs to be runtime compiled
    if ((should_always_recompile() && !array_op_compiler.is_loaded(hash)) ||
        !array_op_compiler.load(hash)) {
        DALI_SCOPE("compilation");
        auto code_template = get_code_template(
            device,
            symbol_table,
            node_to_info
        );
        array_op_compiler.compile<void**, const int*, const int**, const int**, const void**, const int**>(
            hash,
            code_template,
            device.type()
        );
    }
    // return the operation that was loaded or compiled:
    return array_op_compiler.get_function<void**, const int*, const int**, const int**, const void**, const int**>(hash);
}

std::string JITRunner::name() const {
    return "JIT[" + root_.full_expression_name() + "]";
}

Array jit_root(const Array& array) {
    if (is_jit_runner(array)) {
        return as_jit_runner(array)->root_;
    }
    return array;
}

std::tuple<Array, Array> replace_assign_with_inplace(const Array& node) {
    auto assign = static_as_assignment(node);
    auto rightside = jit_root(assign->right());
    auto operator_t = assign->operator_t_;
    if (operator_t == OPERATOR_T_EQL) {
        // in cases where assignment was an implicit cast, ensure that type gets
        // assigned to new destination
        if (rightside.dtype() != node.dtype()) {
            rightside = op::astype(rightside, node.dtype());
        }
        if (rightside.is_scalar() && !node.is_scalar()) {
            rightside = op::jit::tile_scalar(rightside, node.shape());
        }
        return std::tuple<Array, Array>(rightside, Array());
    } else if (operator_t == OPERATOR_T_ADD) {
        return std::tuple<Array, Array>(op::add(assign->left(), rightside), assign->left());
    } else if (operator_t == OPERATOR_T_SUB) {
        return std::tuple<Array, Array>(op::subtract(assign->left(), rightside), assign->left());
    } else if (operator_t == OPERATOR_T_MUL) {
        return std::tuple<Array, Array>(op::eltmul(assign->left(), rightside), assign->left());
    } else if (operator_t == OPERATOR_T_DIV) {
        return std::tuple<Array, Array>(op::eltdiv(assign->left(), rightside), assign->left());
    } else {
        throw std::runtime_error(utils::make_message("No way to replace_assign_with_inplace using operator ",
                                                     operator_to_name(operator_t), "."));
    }
}

Array jit_merge(const Array& root) {
    std::vector<Array> leaves;
    auto assign = static_as_assignment(root);
    ASSERT2(assign->left().is_assignable(), utils::make_message(
        "Assignment destination is not assignable (",
        assign->left().full_expression_name(), ")."));
    auto root_buffer = root.buffer_arg();
    ASSERT2(!root_buffer.is_stateless(), utils::make_message(
        "Assignment destination for JIT assignment ", root.full_expression_name(),
        " does not contain a valid buffer destination (check if left side of "
        "the assignment is assignable)."));
    auto root_operator = assign->operator_t_;
    Array left_leaf, replaced;
    std::vector<Array> all_args = right_args(root);
    auto root_args = root_buffer.expression()->arguments();
    all_args.insert(all_args.end(), root_args.begin(), root_args.end());
    for (auto& arg : all_args) {
        if (arg.is_assignment() &&
            is_jit_runner(static_as_assignment(arg)->right())) {
            // grab leaves from existing jit-runner recursively:
            auto extra_leaves = as_jit_runner(static_as_assignment(arg)->right())->arguments_;
            leaves.insert(leaves.end(), extra_leaves.begin(), extra_leaves.end());
            // if the node is an assignment to a buffer, ensure that
            // the assignment op gets included within this op
            // (e.g. by spoofing the assignment and replacing it with
            //  the equivalent JIT op)
            std::tie(replaced, left_leaf) = replace_assign_with_inplace(arg);
            // if the assignment involves using the left-side (e.g.
            // left += right -> left + right), then keep the left node
            // as a dependency leaf:
            if (!left_leaf.is_stateless()) {
                leaves.emplace_back(left_leaf);
            }
            // now that the jitrunners and assignments are gone, connect
            // up the new operation in the graph:
            arg.set_expression(replaced.expression());
        } else if (arg.is_assignment() | arg.is_control_flow()) {
            // detach the assignment subgraph and only keep the left node(bufferview)
            auto leaf_arg = Array();
            leaf_arg.set_expression(arg.expression());
            arg.set_expression(arg.expression()->buffer_arg());
            leaves.emplace_back(leaf_arg);
        } else if (arg.is_assignable() && is_jit_node(arg)) {
            // detach the assignment subgraph and only keep the left node(bufferview)
            auto leaf_arg = Array();
            leaf_arg.set_expression(arg.expression());
            arg.set_expression(only_buffers(arg.expression()));
            leaves.emplace_back(leaf_arg);
        } else {
            // this node is either an assignment, or a buffer,
            // and is needed as an input here:
            leaves.emplace_back(arg);
        }
    }
    auto new_root = assign->right();
    return Array(std::make_shared<Assignment>(
        // keep the original target buffer:
        assign->left(), root_operator,
        // use the merged operation instead
        Array(std::make_shared<JITRunner>(new_root, leaves, root_operator, root_buffer)))
    );
}

// JIT RUNNER-IMPL //
struct JITRunnerImpl : public Computation {
    using Computation::Computation;
    void run() {
        JITRunner* jit_right = static_cast<JITRunner*>(right_.expression().get());
        int desired_computation_rank = std::max(
            min_computation_rank(left_), jit_right->min_computation_rank_
        );
        SymbolTable symbol_table;
        node_to_info_t node_to_info;
        jit_right->compute_node_compilation_info(desired_computation_rank,
                                                 left_.shape(),
                                                 symbol_table,
                                                 &node_to_info);
        auto device = jit_right->preferred_device();
        auto compiled_self = jit_right->compile(device,
                                                symbol_table,
                                                node_to_info);


        auto buffers = symbol_table.collect_buffers(node_to_info);
        auto scalars = symbol_table.collect_scalars(node_to_info);
        auto shapes_vec = symbol_table.collect_shapes(node_to_info);
        std::vector<const int*> shapes;
        std::transform(shapes_vec.begin(), shapes_vec.end(), std::back_inserter(shapes),
                       [](const std::vector<int>& shape) {return shape.data();});

        std::vector<void*> data_ptrs;
        std::vector<int> array_offsets;
        std::vector<const int*> array_shapes;
        std::vector<const int*> array_strides;
        for (auto& buffer : buffers) {
            data_ptrs.push_back(buffer.memory()->mutable_data(device));
            array_offsets.push_back(buffer.offset());
            array_shapes.push_back(buffer.shape().data());
            array_strides.push_back(buffer.strides().data());
        }
        // std::string assign_name;
        // if (Scope::has_observers()) {
        //     assign_name = ExpressionState::full_operation_name();
        // }
        // DALI_SCOPE(assign_name);
        // std::cout << "running " << assign_name << std::endl;
        compiled_self(
            data_ptrs.data(),
            array_offsets.data(),
            array_shapes.data(),
            array_strides.data(),
            scalars.data(),
            shapes.data()
        );
    }
};

int min_computation_rank(const Array& array) {
    if (is_jit_node(array)) {
        return static_as_jit_node(array)->min_computation_rank_;
    }
    return array.strides().empty() ? 1 : array.ndim();
}

void compute_node_compilation_info(const Array& a,
                                   int desired_computation_rank,
                                   const std::vector<int>& desired_computation_shape,
                                   SymbolTable& symbol_table,
                                   node_to_info_t* node_to_info) {
    if (a.is_buffer()) {
        buffer_compute_node_compilation_info(a, a,
                                             desired_computation_rank,
                                             desired_computation_shape,
                                             symbol_table,
                                             node_to_info);
    } else if (is_jit_node(a)) {
        static_as_jit_node(a)->compute_node_compilation_info(
            desired_computation_rank,
            desired_computation_shape,
            symbol_table,
            node_to_info);
    } else {
        throw std::runtime_error(utils::make_message(
            "Can only compute node compilation info for JITNode "
            "or BufferView (got ", a.expression_name(), ")."));
    }
}


std::string buffer_get_call_code_nd(const Array& array,
                                    const SymbolTable& symbol_table,
                                    const node_to_info_t& node_to_info,
                                    memory::DeviceT device_type) {
    return symbol_table.get_name(array.expression().get());
}


std::string get_call_code_nd(const Array& a,
                             const SymbolTable& symbol_table,
                             const node_to_info_t& node_to_info,
                             memory::DeviceT device_type) {
    if (a.is_buffer()) {
        return buffer_get_call_code_nd(a, symbol_table, node_to_info, device_type);
    } else if (is_jit_node(a)) {
        return static_as_jit_node(a)->get_call_code_nd(symbol_table, node_to_info, device_type);
    } else {
        throw std::runtime_error(utils::make_message(
            "Can only create call code for JITNode "
            "or BufferView (got ", a.expression_name(), ")."));
    }
}

int registered_opt = register_optimization(is_jit_assignment, jit_merge, "jit_merge");
int registered_impl = register_implementation(
   typeid(JITRunner).name(),
   [](Array dest, OPERATOR_T operator_t, Array x, Array assignment) -> std::shared_ptr<Computation> {
        return std::make_shared<JITRunnerImpl>(dest, operator_t, x, assignment);
   });
int registered_buffer = register_implementation(
    typeid(BufferView).name(),
    [](Array dest, OPERATOR_T operator_t, Array x, Array assignment) -> std::shared_ptr<Computation> {
        Array runner(std::make_shared<JITRunner>(
            op::identity(x), std::vector<Array>({x}), operator_t, dest
        ));
        return std::make_shared<JITRunnerImpl>(dest, operator_t, runner, assignment);
    });
int registered_control_flow = register_implementation(
    typeid(ControlFlow).name(),
    [](Array dest, OPERATOR_T operator_t, Array x, Array assignment) -> std::shared_ptr<Computation> {
        Array runner(std::make_shared<JITRunner>(
            op::identity(x.buffer_arg()), std::vector<Array>({x}), operator_t, dest
        ));
        return std::make_shared<JITRunnerImpl>(dest, operator_t, runner, assignment);
    });
int registered_assignment = register_implementation(
    typeid(Assignment).name(),
    [](Array dest, OPERATOR_T operator_t, Array x, Array assignment) -> std::shared_ptr<Computation> {
        Array runner(std::make_shared<JITRunner>(
            op::identity(x.buffer_arg()), std::vector<Array>({x}), operator_t, dest
        ));
        return std::make_shared<JITRunnerImpl>(dest, operator_t, runner, assignment);
    });
}  // namespace jit
}  // namespace op
