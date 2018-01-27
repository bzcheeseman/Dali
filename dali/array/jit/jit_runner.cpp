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
    auto pos = arrays_visited_.find(ptr);
    if (pos == arrays_visited_.end()) {
        arrays_.emplace_back(ptr);
        int index = arrays_visited_.size();
        arrays_visited_.emplace(ptr, index);
        array_order_.add(index);
    } else {
        array_order_.add(pos->second);
    }
}

int SymbolTable::get_array_index(const BufferView* ptr) const {
    return arrays_visited_.at(ptr);
}

int SymbolTable::get_scalar_index(const ScalarView* ptr) const {
    return scalars_visited_.at(ptr);
}

void SymbolTable::declare_scalar(const ScalarView* ptr) {
    auto pos = scalars_visited_.find(ptr);
    if (pos == scalars_visited_.end()) {
        scalars_.emplace_back(ptr);
        int index = scalars_visited_.size();
        scalars_visited_.emplace(ptr, index);
        scalar_order_.add(index);
    } else {
        scalar_order_.add(pos->second);
    }
}

namespace {
    hash_t node_temporary_hash(const CompilationInfo& info) {
        return utils::Hasher().add(info.hash).add(info.data_hash).value();
    }
}
// TODO(jonathan): ensure declare_shape is not needed, and shape presence can be
// created based on get_call_code and nothing else.
void SymbolTable::declare_shape(const Expression* ptr) {
    shapes_.emplace_back(ptr);
}


void SymbolTable::store_into_temporary(expression_ptr ptr, node_to_info_t& node_to_info) {
    // TODO(jonathan): some of the work in this function can be procastinated until
    // the compilation stage (and thus only run once).
    if (!ptr->is_buffer() && static_cast<JITNode*>(ptr.get())->antialias()) {
        auto pos = node_to_info.find(ptr.get());
        ASSERT2(pos != node_to_info.end(), utils::make_message(
            "store_into_temporary called before compute_node_compilation_info was run on Expression ",
            ptr->full_name(), ".\nCall store_into_temporary after compute_node_compilation_info."));
        auto node_hash = node_temporary_hash(pos->second);
        auto node_pos = std::find(temporary_assigns_expression_hashes_.begin(),
                                  temporary_assigns_expression_hashes_.end(),
                                  node_hash);
        if (node_pos == temporary_assigns_expression_hashes_.end()) {
            temporary_assigns_expressions_.emplace_back(ptr);
            temporary_assigns_expression_hashes_.emplace_back(node_hash);
            Array temp(ptr->shape_.size() > 0 ? ptr->shape_ : std::vector<int>({1,}), ptr->dtype_, ptr->preferred_device());
            temporaries_.emplace_back(temp);
            node_to_info[temp.expression().get()].computation_rank = pos->second.computation_rank;
        }
    }
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
        ".\nDon't forget to call `shape_required()` return true."));
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
        shape_declaration_table_[shapes_[i]] = name;
        result << build_shape_definition(
            name,
            node_to_info.at(shapes_[i]).computation_rank,
            utils::make_message("shapes[", i, "]")
        );
    }

    for (int i = 0; i < temporaries_.size(); ++i) {
        auto ptr = temporaries_[i].expression().get();
        auto name = utils::make_message("temp_", i);
        declaration_table_[ptr] = name;
        result << build_array_definition(
            dtype_to_cpp_name(ptr->dtype_),
            name,
            true,
            node_to_info.at(ptr).computation_rank,
            utils::make_message(
                "array_data[", arrays_.size() + i, "], "
                "offsets[", arrays_.size() + i, "], "
                "sizes[", arrays_.size() + i, "]"
            )
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
    arrays.insert(arrays.end(), temporaries_.begin(), temporaries_.end());
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

JITNode::JITNode(const std::vector<int>& shape,
                 DType dtype,
                 const std::vector<Array>& arguments,
                 int offset,
                 const std::vector<int>& strides) : Expression(shape, dtype, arguments, offset, strides) {}

JITNode::JITNode(const JITNode& other) : Expression(other) {};

std::string JITNode::prefix_code(const node_to_info_t& node_to_info, memory::DeviceT device_type) const {
    return "";
}

bool JITNode::shape_required() const {
    return false;
}

bool JITNode::antialias() const {
    return true;
}

bool JITNode::is_axis_collapsible_with_axis_minus_one(int axis) const {
    return false;
}

expression_ptr JITNode::_reshape(const std::vector<int>& new_shape, const Array*) const {
    // customize the way reshapes get run on JIT-nodes:
    // run them in place without a copy.
    return op::jit::jit_view(copy(), new_shape, 0, {}).expression();
}

expression_ptr JITNode::_expand_dims(int axis, const Array*) const {
    // customize the way reshapes get run on JIT-nodes:
    // run them in place without a copy.
    return op::jit::expand_dims(copy(), axis).expression();
}

expression_ptr JITNode::_squeeze(int axis, const Array*) const {
    // customize the way reshapes get run on JIT-nodes:
    // run them in place without a copy.
    return op::jit::squeeze(copy(), axis).expression();
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

PARALLELISM_T JITNode::parallelism_type() const {
    for (auto& arg : arguments()) {
        // if any of the elements below this node is
        // not acting independent of the warp, then assume
        // the warp needs to stay as is
        auto ptype = op::jit::parallelism_type(arg);
        if (ptype != INDEPENDENT_BLOCK_WARP) {
            return ptype;
        }
    }
    return INDEPENDENT_BLOCK_WARP;
}

hash_t JITNode::compute_node_data_hash(const node_to_info_t& node_to_info, const SymbolTable& symbol_table) const {
    utils::Hasher hasher;
    for (const auto& arg : arguments_) {
        hasher.add(node_to_info.at(arg.expression().get()).data_hash);
    }
    for (auto& val : shape_) {
        hasher.add(val);
    }
    return hasher.value();
}

bool JITNode::supports_operator(OPERATOR_T operator_t) const {
    return true;
}

std::string JITNode::assignment_code_nd(OPERATOR_T operator_t, memory::DeviceT device_type,
                                        std::string dst, std::string src) const {
    return utils::make_message(dst, " ", operator_to_name(operator_t), " ", src);
}

std::string JITNode::assignment_prefix_code(hash_t hash,
                                            const std::vector<OPERATOR_T>& operators,
                                            const node_to_info_t& node_to_info,
                                            memory::DeviceT device_type,
                                            const std::vector<int>& computation_ranks,
                                            const std::vector<PARALLELISM_T>& parallelism_types) const {
    if (device_type == memory::DEVICE_T_GPU) {
        std::stringstream ss;
        ss << "template <";
        for (int i = 0; i < computation_ranks.size(); i++) {
            ss << "typename Destination" << i << ", typename Source" << i;
            if (i + 1 != computation_ranks.size()) {
                ss << ", ";
            }
        }
        ss << ">\n"
              "void __global__\n"
              "assign_kernel_" << hash << "(";
        for (int i = 0; i < computation_ranks.size(); i++) {
            ss << "Destination" << i << " dst" << i
               << ", Source" << i << " src" << i;
            if (parallelism_types[i] == INDEPENDENT_BLOCK_WARP) {
               ss << ", int num_el" << i;
            }
            if (computation_ranks[i] > 1) {
                ss << ", Shape<" << computation_ranks[i] << "> shape" << i;
            }
            if (i + 1 != computation_ranks.size()) {
                ss << ", ";
            } else {
                ss << ") {\n";
            }
        }
        ss << "    int stride = blockDim.x * gridDim.x;\n";

        std::unordered_map<int, std::string> nd_idxes;

        for (int i = 0; i < computation_ranks.size(); i++) {
            // choose whether threads in a warp are part of the indexing structure.
            if (parallelism_types[i] == INDEPENDENT_BLOCK_WARP) {
                ss << "    " << (i == 0 ? "int " : "") << "idx = ";
                ss << "blockDim.x * blockIdx.x + threadIdx.x;\n";
                if (computation_ranks[i] == 1) {
                    ss << utils::make_message(
                        "    for (int i = idx; i < num_el", i, "; i += stride) {\n"
                        "        ", assignment_code_nd(operators[i], device_type,
                                        utils::make_message("dst", i, "[i]"),
                                        utils::make_message("src", i, "[i]")
                                    ), ";\n"
                        "    }\n"
                    );
                } else {
                    ss << utils::make_message(
                        "    for (int i = idx; i < num_el", i, "; i += stride) {\n"
                        "        auto nd_idx = index_to_dim(idx, shape", i, ");\n"
                        "        ", assignment_code_nd(operators[i], device_type,
                                        utils::make_message("dst", i, "[nd_idx]"),
                                        utils::make_message("src", i, "[nd_idx]")
                                    ), ";\n"
                        "    }\n"
                    );
                }
            } else {
                if (i == 0 || parallelism_types[i - 1] != parallelism_types[i]) {
                    ss << "    " << (i == 0 ? "int " : "") << "idx = blockIdx.x;\n";
                }
                std::string query;
                if (computation_ranks[i] > 1) {
                    std::string nd_idx_name;
                    ss << "    ";
                    if (nd_idxes.find(computation_ranks[i]) != nd_idxes.end()) {
                        nd_idx_name = nd_idxes.at(computation_ranks[i]);
                    } else {
                        nd_idx_name = utils::make_message("nd_idx_", computation_ranks[i]);
                        ss << "auto ";
                        nd_idxes[computation_ranks[i]] = nd_idx_name;
                    }
                    ss << nd_idx_name << " = index_to_dim(idx, shape" << i << ");\n";
                    query = utils::make_message("[", nd_idx_name, "]");
                } else {
                    query = "[idx]";
                }
                ss << "    " << assignment_code_nd(operators[i], device_type,
                            utils::make_message("dst", i, query),
                            utils::make_message("src", i, query)) << ";\n";
            }
            if (i + 1 != computation_ranks.size()) {
                ss << "    __syncthreads();\n";
            }
        }
        ss << "}\n";
        return ss.str();
    }
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

    virtual expression_ptr copy() const override;
    JITRunner(Array root, const std::vector<Array>& leaves, OPERATOR_T operator_t, Array dest);
    virtual memory::Device preferred_device() const override;
    virtual void compute_node_compilation_info(int desired_computation_rank,
                                               const std::vector<int>& desired_computation_shape,
                                               SymbolTable& symbol_table,
                                               node_to_info_t& node_to_info) const override;
    virtual bool is_axis_collapsible_with_axis_minus_one(int axis) const override;
    virtual std::string get_call_code_nd(const SymbolTable& symbol_table,
                                         const node_to_info_t& node_to_info,
                                         memory::DeviceT device_type) const override;

    virtual expression_ptr jit_right_fit_ndim(int ndim) const override {
        return std::make_shared<JITRunner>(
            op::jit::jit_right_fit_ndim(root_, ndim),
            arguments_,
            operator_t_,
            op::jit::jit_right_fit_ndim(dest_, ndim));
    }

    virtual int min_computation_rank() const override;

    std::string get_code_template(hash_t hash,
                                  memory::Device device,
                                  const SymbolTable& symbol_table,
                                  const node_to_info_t& node_to_info) const;

    std::vector<Assignment> create_assignment_sequence(
        const SymbolTable& symbol_table,
        const node_to_info_t& node_to_info) const;

    std::function<void(void**, const int*, const int**, const int**, const void**, const int**)> compile(
        memory::Device,
        const SymbolTable& symbol_table,
        const node_to_info_t& node_to_info) const;
    virtual std::string name() const override;
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
        JITNode(dest.shape(), root.dtype(), leaves),
        root_(root), operator_t_(operator_t), dest_(dest) {}

int JITRunner::min_computation_rank() const {
    return std::max(op::jit::min_computation_rank(dest_), as_jit_node(root_)->min_computation_rank());
}

expression_ptr JITRunner::copy() const {
    return std::make_shared<JITRunner>(*this);
}

memory::Device JITRunner::preferred_device() const {
    return root_.preferred_device();
}

bool JITRunner::is_axis_collapsible_with_axis_minus_one(int axis) const {
    return static_as_jit_node(root_)->is_axis_collapsible_with_axis_minus_one(axis);
}

static hash_t BUFFER_HASH = std::hash<std::string>()(typeid(BufferView).name());


void buffer_compute_node_compilation_info(const Array& array, const Array& buffer_array,
                                          int desired_computation_rank,
                                          const std::vector<int>& desired_computation_shape,
                                          SymbolTable& symbol_table,
                                          node_to_info_t& node_to_info) {
    const BufferView* buffer_ptr = static_cast<const BufferView*>(buffer_array.expression().get());
    symbol_table.declare_array(buffer_ptr);
    node_to_info[buffer_ptr].computation_rank  = desired_computation_rank;
    node_to_info[buffer_ptr].computation_shape = desired_computation_shape;
    node_to_info[buffer_ptr].hash = utils::Hasher().add(BUFFER_HASH)
                                                   .add(desired_computation_rank)
                                                   .add(buffer_requires_strides(buffer_ptr, desired_computation_shape))
                                                   .add(array.dtype()).value();
    node_to_info[buffer_ptr].data_hash = symbol_table.get_array_index(buffer_ptr);
}



void subexpression_elimination(const Array& root,
                               node_to_info_t& node_to_info,
                               SymbolTable& symbol_table,
                               std::unordered_map<hash_t, int>& occurence_map) {
    hash_t node_hash = node_temporary_hash(node_to_info.at(root.expression().get()));
    occurence_map[node_hash] += 1;
    // if it's worthwhile (basically) -- e.g. test for expensiveness here
    if (occurence_map[node_hash] > 1) {
        symbol_table.store_into_temporary(root.expression(), node_to_info);
    } else {
        for (const auto& arg : root.expression()->arguments()) {
            subexpression_elimination(arg, node_to_info, symbol_table, occurence_map);
        }
    }
}

void subexpression_elimination(const Array& root,
                               node_to_info_t& node_to_info,
                               SymbolTable& symbol_table) {
    std::unordered_map<hash_t, int> occurence_map;
    subexpression_elimination(root, node_to_info, symbol_table, occurence_map);
}


void recursive_declare_shape(const Array& root,
                             node_to_info_t& node_to_info,
                             SymbolTable& symbol_table) {
    if (is_jit_node(root) && static_as_jit_node(root)->shape_required()) {
        symbol_table.declare_shape(root.expression().get());
    }
    for (const auto& arg : root.expression()->arguments()) {
        recursive_declare_shape(arg, node_to_info, symbol_table);
    }
}


void JITRunner::compute_node_compilation_info(int desired_computation_rank,
                                              const std::vector<int>& desired_computation_shape,
                                              SymbolTable& symbol_table,
                                              node_to_info_t& node_to_info) const {
    node_to_info[this].computation_rank = desired_computation_rank;

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

    // look for repeated computation opportunities?
    subexpression_elimination(root_, node_to_info, symbol_table);
    recursive_declare_shape(root_, node_to_info, symbol_table);
    recursive_declare_shape(dest_, node_to_info, symbol_table);

    utils::Hasher hasher;
    hasher.add(optype_hash)
          .add(desired_computation_rank)
          .add(node_to_info.at(dest_.expression().get()).hash)
          .add(node_to_info.at(root_.expression().get()).hash)
          .add(int(operator_t_));
    node_to_info[this].hash = hasher.value();
}

std::string JITRunner::get_call_code_nd(const SymbolTable& symbol_table,
                                        const node_to_info_t& node_to_info,
                                        memory::DeviceT device_type) const {
    return "";
}

std::string JITNode::assignment_code(hash_t hash,
                                     const std::vector<Array>& dests,
                                     const std::vector<std::string>& roots,
                                     const std::vector<OPERATOR_T>& operators,
                                     const SymbolTable& symbol_table,
                                     const node_to_info_t& node_to_info,
                                     memory::DeviceT device_type,
                                     const std::vector<int>& computation_ranks,
                                     const std::vector<PARALLELISM_T>& parallelism_types) const {


    std::vector<std::string> dest_call_codes;
    for (const auto& dest : dests) {
        dest_call_codes.emplace_back(
            op::jit::get_call_code_nd(dest, symbol_table, node_to_info, device_type)
        );
    }
    if (device_type == memory::DEVICE_T_CPU) {
        std::stringstream ss;
        ss << "    int num_el;\n";
        for (int i = 0; i < computation_ranks.size(); i++) {
            int computation_rank = computation_ranks[i];
            std::string indexing_nd = computation_rank == 1 ? "[i]" : "[" + generate_accessor_string(computation_rank) + "]";
            std::string assign_line = assignment_code_nd(
                operators[i], device_type, utils::make_message(dest_call_codes[i], indexing_nd),
                utils::make_message(roots[i], indexing_nd));
            if (computation_rank == 1) {
                ss << utils::make_message(
                    "    num_el = ", dest_call_codes[i], ".shape().numel();\n",
                    "    #pragma clang loop vectorize(enable)\n",
                    "    #pragma clang loop interleave(enable)\n",
                    "    for (int i = 0; i < num_el; ++i) {\n",
                    "        ", assign_line, ";\n"
                    "    }\n"
                );
            } else {
                ss << construct_for_loop(
                    computation_rank, assign_line, dest_call_codes[i], 4);
            }
        }
        return ss.str();
    } else if (device_type == memory::DEVICE_T_GPU) {
        std::stringstream ss;
        ss << "    const int NT = 256;\n"
              "    // const int MAX_BLOCKS = 40960;\n";
        for (int i = 0; i < computation_ranks.size(); i++) {
            ss << "    auto dest_shape_" << i << " = " << dest_call_codes[i] << ".shape();\n";
            ss << "    int num_el" << i << " = dest_shape_" << i << ".numel();\n";
            if (computation_ranks.size() > 1) {
                if (i == 0) {
                    ss << "    int max_num_el = num_el0;\n";
                } else {
                    ss << "    max_num_el = max(num_el" << i << ", max_num_el);\n";
                }
            }
        }
        ss << "    int grid_size = ";
        if (std::any_of(parallelism_types.begin(),
                        parallelism_types.end(),
                        [](PARALLELISM_T ptype) {return ptype == INDEPENDENT_BLOCK;})) {
            ss << (computation_ranks.size() == 1 ? "num_el0;\n" : "max_num_el;\n");
        } else {
            ss << "div_ceil(" << (computation_ranks.size() == 1 ? "num_el0" : "max_num_el") << ", NT);\n";
        }
        ss << "    // assert(grid_size <= MAX_BLOCKS);\n"
              "    assign_kernel_" << hash << "<<<grid_size, NT, 0, NULL>>>(\n";
        for (int i = 0; i < computation_ranks.size(); i++) {
            int computation_rank = computation_ranks[i];
            ss << "        " << dest_call_codes[i] << ",\n"
               << "        " << roots[i];
            if (parallelism_types[i] == INDEPENDENT_BLOCK_WARP) {
                ss << ",\n        num_el" << i;
            }
            if (computation_rank > 1) {
                ss << ",\n";
                ss << "        dest_shape_" << i;
            }
            if (i + 1 < computation_ranks.size()) {
                ss << ",";
            }
            ss << "\n";
        }
        ss << "    );\n";
        return ss.str();
    } else {
        ASSERT2(false, "unknown device type.");
    }
}

expression_ptr JITNode::jit_right_fit_ndim(int rank) const {
    ASSERT2(false, utils::make_message(
        full_name(), " node declared to have a min computation "
        "rank (", min_computation_rank(), ") lower than its ndim (", ndim(),
        "). Override `jit_right_fit_ndim` or declare the node with "
        "min_computation_rank == ndim."));
    return nullptr;
}

void rewrite_with_temporaries(const Array& root,
                              const std::unordered_map<hash_t, Array>& temps,
                              const node_to_info_t& node_to_info) {
    if (temps.size() == 0) {
        return;
    }
    for (const auto& arg : root.expression()->arguments()) {
        auto found_temp = temps.find(node_temporary_hash(node_to_info.at(arg.expression().get())));
        if (found_temp != temps.end()) {
            arg.set_expression(found_temp->second.expression());
        } else {
            rewrite_with_temporaries(arg, temps, node_to_info);
        }
    }
}

std::vector<Assignment> JITRunner::create_assignment_sequence(
        const SymbolTable& symbol_table,
        const node_to_info_t& node_to_info) const {
    std::vector<Assignment> assignments;
    std::unordered_map<hash_t, Array> computed_temporaries;
    for (size_t i = 0; i < symbol_table.temporaries_.size(); i++) {
        auto expr_ptr = symbol_table.temporary_assigns_expressions_[i];
        Array new_right(expr_ptr);
        rewrite_with_temporaries(new_right, computed_temporaries, node_to_info);
        assignments.emplace_back(
            symbol_table.temporaries_[i],
            OPERATOR_T_EQL,
            new_right
        );
        // notify future template generation steps that a particular node
        // no longer needs to be computed
        computed_temporaries[node_temporary_hash(node_to_info.at(expr_ptr.get()))] = symbol_table.temporaries_[i];
    }
    rewrite_with_temporaries(root_, computed_temporaries, node_to_info);
    assignments.emplace_back(dest_, operator_t_, root_);
    return assignments;
}

namespace {
    struct JITOptimization {
        typedef std::function<bool(const Array&, memory::DeviceT, const node_to_info_t&)> condition_t;
        condition_t condition_;
        int priority_;
        std::function<Array(const Array&)> transformation_;
        std::string name_;
        bool matches(const Array& array, memory::DeviceT device_type, const node_to_info_t& node_to_info) const {
            return condition_(array, device_type, node_to_info);
        }
        Array transform(const Array& array) const {
            return transformation_(array);
        }
        JITOptimization(int priority,
                        condition_t condition,
                        std::function<Array(const Array&)> transformation,
                        const std::string& name) :
            priority_(priority), condition_(condition),
            transformation_(transformation), name_(name) {}
    };

    std::vector<JITOptimization> JIT_OPTIMIZATIONS;
    void recursive_jit_optimize(const Array& root, memory::DeviceT device_type, node_to_info_t& node_to_info) {
        if (!is_jit_node(root)) return;
        for (auto& arg : root.expression()->arguments()) {
            recursive_jit_optimize(arg, device_type, node_to_info);
        }

        for (const auto& optimization : JIT_OPTIMIZATIONS) {
            if (optimization.matches(root, device_type, node_to_info)) {
                auto new_root = optimization.transform(root);
                // guard optimization behavior so that shapes are preserved
                ASSERT2(new_root.shape() == root.shape(), utils::make_message(
                    "JIT Optimization '", optimization.name_, "' altered the shape of the operation"
                    " from ", root.shape(), " to ", new_root.shape(),
                    " on expression ", root.full_expression_name(), "."
                    ));
                ASSERT2(new_root.dtype() == root.dtype(), utils::make_message(
                    "JIT Optimization '", optimization.name_, "' altered the dtype of the operation"
                    " from ", dtype_to_name(root.dtype()), " to ", dtype_to_name(new_root.dtype()),
                    " on expression ", root.full_expression_name(), "."
                    ));
                node_to_info[new_root.expression().get()] = node_to_info.at(root.expression().get());
                root.set_expression(new_root.expression());
            }
        }
    }
}

int register_jit_optimization(int priority,
                              JITOptimization::condition_t condition,
                              std::function<Array(const Array&)> transformation,
                              const std::string& name) {
    JIT_OPTIMIZATIONS.emplace_back(priority, condition, transformation, name);
    std::sort(JIT_OPTIMIZATIONS.begin(), JIT_OPTIMIZATIONS.end(),
              [](const JITOptimization& left, const JITOptimization& right) {return left.priority_ < right.priority_;});
    return 0;
}

std::string JITRunner::get_code_template(hash_t hash,
                                         memory::Device device,
                                         const SymbolTable& symbol_table,
                                         const node_to_info_t& node_to_info) const {
    auto assignments = create_assignment_sequence(symbol_table, node_to_info);
    // TODO(jonathan): ensure that ops that are effectively buffer view
    // reshapes/restrides/etc.. are done directly on the temporaries (if possible)
    // (less hot loop code)

    // 1. TODO(jonathan): now rewrite such that a parallelism structure is chosen for each
    // op in a tower (e.g. by querying an internal function for JITNodes...)
    auto new_node_to_info = node_to_info;
    for (const auto& assignment : assignments) {
        recursive_jit_optimize(assignment.right(), device.type(), new_node_to_info);
    }
    // move up...
    // continue...
    // optimized_assignments = ...
    //
    // 2. now generate code with optimized assignments

    std::unordered_set<hash_t> prefix_code_visited;
    std::stringstream result;
    auto insert_once = [&](const std::string& pc) {
        auto pc_hash = utils::get_hash(pc);
        if (prefix_code_visited.find(pc_hash) == prefix_code_visited.end()) {
            result << pc;
            prefix_code_visited.insert(pc_hash);
        }
    };
    auto add_prefix_code = [&](const Array& arr) {
        if (is_jit_node(arr)) {
            insert_once(static_as_jit_node(arr)->prefix_code(new_node_to_info, device.type()));
        }
    };

    std::vector<int> computation_ranks;
    std::vector<OPERATOR_T> operators;
    std::vector<PARALLELISM_T> parallelism_types;

    for (const auto& assignment : assignments) {
        const auto& node_info = new_node_to_info.at(assignment.left().expression().get());
        computation_ranks.emplace_back(node_info.computation_rank);
        parallelism_types.emplace_back(op::jit::parallelism_type(assignment.right()));
        operators.emplace_back(assignment.operator_t_);
        assignment.left().expression()->for_all_suboperations(add_prefix_code);
        add_prefix_code(assignment.right());
        assignment.right().expression()->for_all_suboperations(add_prefix_code);
    }

    if (!assignments.back().left().is_buffer()) {
        insert_once(as_jit_node(assignments.back().left())->assignment_prefix_code(
            hash, operators, new_node_to_info, device.type(),
            computation_ranks, parallelism_types));
    } else {
        insert_once(assignment_prefix_code(
            hash, operators, new_node_to_info, device.type(),
            computation_ranks, parallelism_types));
    }

    result << "void run(void** array_data, const int* offsets, "
              "const int** sizes, const int** strides, "
              "const void** scalar_arguments, const int** shapes) {\n";
    // DECLARE SYMBOLS
    result << symbol_table.variable_declarations(new_node_to_info) << "\n";

    std::vector<std::string> roots;
    std::vector<Array> dests;
    for (const auto& assign : assignments) {
        roots.emplace_back(
            static_as_jit_node(assign.right())->get_call_code_nd(
                symbol_table, new_node_to_info, device.type()
            ));
        dests.emplace_back(assign.left());
    }

    if (!assignments.back().left().is_buffer()) {
        result << as_jit_node(assignments.back().left())->assignment_code(hash, dests, roots, operators, symbol_table, new_node_to_info, device.type(),
                                                      computation_ranks, parallelism_types);
    } else {
        result << assignment_code(hash, dests, roots, operators, symbol_table, new_node_to_info, device.type(),
                                  computation_ranks, parallelism_types);
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
                                 .add(symbol_table.array_order_.value())
                                 .add(symbol_table.scalar_order_.value())
                                 .value();
    // check if the operation needs to be runtime compiled
    if ((should_always_recompile() && !array_op_compiler.is_loaded(hash)) ||
        !array_op_compiler.load(hash)) {
        DALI_SCOPE("compilation");
        auto code_template = get_code_template(hash,
                                               device,
                                               symbol_table,
                                               node_to_info);
        array_op_compiler.compile<void**, const int*, const int**, const int**, const void**, const int**>(
            hash,
            code_template,
            device.type(),
            ""
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
        int desired_computation_rank = jit_right->min_computation_rank();
        SymbolTable symbol_table;
        node_to_info_t node_to_info;
        auto right = right_.expression();

        if (desired_computation_rank < jit_right->ndim()) {
            right = jit_right->jit_right_fit_ndim(desired_computation_rank);
            jit_right = static_cast<JITRunner*>(right.get());
        }

        jit_right->compute_node_compilation_info(desired_computation_rank,
                                                 jit_right->dest_.shape(),
                                                 symbol_table,
                                                 node_to_info);

        auto device = jit_right->preferred_device();
        auto buffers = symbol_table.collect_buffers(node_to_info);
        auto scalars = symbol_table.collect_scalars(node_to_info);
        auto shapes_vec = symbol_table.collect_shapes(node_to_info);
        auto compiled_self = jit_right->compile(device,
                                                symbol_table,
                                                node_to_info);
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

expression_ptr jit_right_fit_ndim(const Array& array, int ndim) {
    if (array.ndim() <= 1 | array.ndim() == ndim) return array.expression();
    if (array.is_buffer()) {
        return array.right_fit_ndim(ndim).expression();
    }
    return static_as_jit_node(array)->jit_right_fit_ndim(ndim);
}

int min_computation_rank(const Array& array) {
    if (is_jit_node(array)) {
        int rank = static_as_jit_node(array)->min_computation_rank();
        ASSERT2(rank > 0, utils::make_message(
            array.full_expression_name(), " computation rank must be greater than 0 (got "
            "min_computation_rank = ", min_computation_rank, ")."));
        return rank;
    }
    return array.strides().empty() ? 1 : array.ndim();
}

PARALLELISM_T parallelism_type(const Array& array) {
    if (is_jit_node(array)) {
        return static_as_jit_node(array)->parallelism_type();
    }
    return INDEPENDENT_BLOCK_WARP;
}

void compute_node_compilation_info(const Array& a,
                                   int desired_computation_rank,
                                   const std::vector<int>& desired_computation_shape,
                                   SymbolTable& symbol_table,
                                   node_to_info_t& node_to_info) {
    if (a.is_buffer()) {
        buffer_compute_node_compilation_info(a, a,
                                             desired_computation_rank,
                                             desired_computation_shape,
                                             symbol_table,
                                             node_to_info);
    } else if (is_jit_node(a)) {
        auto node = static_as_jit_node(a);
        node->compute_node_compilation_info(
            desired_computation_rank,
            desired_computation_shape,
            symbol_table,
            node_to_info);
        node_to_info[node].data_hash = node->compute_node_data_hash(node_to_info, symbol_table);
        // TODO(jonathan): ask here if this node has been computed beforehand some
        // number of times, and whether it is worth recomputing it, or using it
        // from some previously stored location.
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
