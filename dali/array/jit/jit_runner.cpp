#include "jit_runner.h"

#include <unordered_set>

#include "dali/utils/make_message.h"
#include "dali/utils/assert2.h"
#include "dali/array/expression/optimization.h"
#include "dali/array/expression/assignment.h"
#include "dali/array/expression/computation.h"
#include "dali/array/jit/jit_utils.h"
#include "dali/utils/compiler.h"
#include "dali/utils/scope.h"
#include "dali/array/op/binary.h"
#include "dali/array/op/unary.h"

namespace op {
namespace jit {

bool should_always_recompile_is_cached = false;
bool should_always_recompile_cache     = false;

bool should_always_recompile() {
    if (!should_always_recompile_is_cached) {
        auto env_var_ptr = std::getenv("DALI_RTC_ALWAYS_RECOMPILE");
        std::string dali_rtc_always_recompile;
        if (env_var_ptr == NULL) {
            dali_rtc_always_recompile = "false";
        } else {
            dali_rtc_always_recompile = env_var_ptr;
        }

        // lower
        for (int i = 0; i < dali_rtc_always_recompile.size(); ++i) {
            if ('A' <= dali_rtc_always_recompile[i] && dali_rtc_always_recompile[i] <= 'Z') {
                dali_rtc_always_recompile[i] += 'a' - 'A';
            }
        }
        should_always_recompile_cache = (dali_rtc_always_recompile == "true");
        should_always_recompile_is_cached = true;
    }
    return should_always_recompile_cache;
}

// CONVENIENCE METHODS //
std::shared_ptr<JITNode> as_jit_node(Array array) {
    auto casted = std::dynamic_pointer_cast<JITNode>(array.expression());
    ASSERT2(casted != nullptr, utils::make_message("Attempting to cast a non-jit expression (",
        array.expression_name(), ") into a JITNode."));
    return casted;
}

hash_t node_hash(const node_to_info_t& node_to_info, const Array& array) {
    return node_to_info.at(array.expression().get()).hash;
}

// JIT NODE
JITNode::JITNode(int min_computation_rank,
                 const std::vector<int>& shape,
                 DType dtype,
                 int offset,
                 const std::vector<int>& strides) : Expression(shape, dtype, offset, strides),
                                                    min_computation_rank_(min_computation_rank) {
    ASSERT2(min_computation_rank > 0, utils::make_message(
        "JITNode computation rank must be greater than 0."));
}

JITNode::JITNode(const JITNode& other) : Expression(other),
                                         min_computation_rank_(other.min_computation_rank_) {};


std::string JITNode::prefix_code(const node_to_info_t& node_to_info, memory::DeviceT device_type) const {
    return "";
}

bool JITNode::is_axis_collapsible_with_axis_minus_one(const int& axis) const {
    return false;
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

struct JITRunner : public JITNode {
    static const hash_t optype_hash;
    Array dest_;
    Array root_;
    std::vector<Array> leaves_;
    const OPERATOR_T operator_t_;

    virtual expression_ptr copy() const;
    JITRunner(Array root, const std::vector<Array>& leaves, OPERATOR_T operator_t, Array dest);
    virtual std::vector<Array> arguments() const;
    virtual memory::Device preferred_device() const;
    virtual std::string prefix_code(const node_to_info_t& node_to_info,
                                    memory::DeviceT device_type) const;
    virtual void compute_node_compilation_info(int desired_computation_rank,
                                               const std::vector<int>& desired_computation_shape,
                                               std::vector<const BufferView*>* arrays,
                                               std::vector<const ScalarView*>* scalars,
                                               node_to_info_t* node_to_info) const;
    virtual bool is_axis_collapsible_with_axis_minus_one(const int& axis) const;
    virtual std::string get_call_code_nd(const symbol_table_t& symbol_table,
                                         const node_to_info_t& node_to_info,
                                         memory::DeviceT device_type) const;

    std::string get_code_template(memory::Device device,
                      const std::vector<const BufferView*>& arrays,
                      const std::vector<const ScalarView*>& scalars,
                      const node_to_info_t& node_to_info) const;

    std::string dest_assignment_code(const symbol_table_t& symbol_table,
                                     const node_to_info_t& node_to_info,
                                     memory::DeviceT device_type) const;
    std::string assignment_code(const symbol_table_t& symbol_table,
                                const node_to_info_t& node_to_info,
                                memory::DeviceT device_type) const;

    std::function<void(void**, const int*, const int**, const int**, const void**)> compile(
        memory::Device,
        const std::vector<const BufferView*>& arrays,
        const std::vector<const ScalarView*>& scalars,
        const node_to_info_t& node_to_info) const;
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
            is_jit_node(as_assignment(node)->right_) &&
            !is_jit_runner(as_assignment(node)->right_));
}

std::shared_ptr<JITRunner> as_jit_runner(const Array& array) {
    return std::dynamic_pointer_cast<JITRunner>(array.expression());
}

// JIT RUNNER //
JITRunner::JITRunner(Array root, const std::vector<Array>& leaves, OPERATOR_T operator_t, Array dest) :
        JITNode(as_jit_node(root)->min_computation_rank_, dest.shape(), root.dtype()),
        root_(root), leaves_(leaves), operator_t_(operator_t), dest_(dest) {
    if (is_jit_runner(root)) {
        throw std::runtime_error("JITRunner should not contain a JITRunner.");
    }
}

std::vector<Array> JITRunner::arguments() const {
    return leaves_;
}
// TODO(jonathan): add pretty-printing here to keep track of what was jitted or not.

expression_ptr JITRunner::copy() const {
    return std::make_shared<JITRunner>(*this);
}

memory::Device JITRunner::preferred_device() const {
    return root_.preferred_device();
}

bool JITRunner::is_axis_collapsible_with_axis_minus_one(const int& axis) const {
    return as_jit_node(root_)->is_axis_collapsible_with_axis_minus_one(axis);
}

static hash_t BUFFER_HASH = std::hash<std::string>()(typeid(BufferView).name());

void buffer_compute_node_compilation_info(const Array& array,
                                          int desired_computation_rank,
                                          const std::vector<int>& desired_computation_shape,
                                          std::vector<const BufferView*>* arrays,
                                          std::vector<const ScalarView*>* scalars,
                                          node_to_info_t* node_to_info) {
    const BufferView* ptr = static_cast<const BufferView*>(array.expression().get());
    arrays->emplace_back(ptr);
    (*node_to_info)[ptr].computation_rank  = desired_computation_rank;
    (*node_to_info)[ptr].computation_shape = desired_computation_shape;
    (*node_to_info)[ptr].hash = utils::Hasher().add(BUFFER_HASH)
                                               .add(desired_computation_rank)
                                               .add(ptr->contiguous_memory())
                                               .add(array.dtype()).value();
}


void JITRunner::compute_node_compilation_info(int desired_computation_rank,
                                              const std::vector<int>& desired_computation_shape,
                                              std::vector<const BufferView*>* arrays,
                                              std::vector<const ScalarView*>* scalars,
                                              node_to_info_t* node_to_info) const {
    (*node_to_info)[this].computation_rank = desired_computation_rank;

    op::jit::compute_node_compilation_info(dest_,
                                           desired_computation_rank,
                                           desired_computation_shape,
                                           arrays,
                                           scalars,
                                           node_to_info);
    op::jit::compute_node_compilation_info(root_,
                                           desired_computation_rank,
                                           desired_computation_shape,
                                           arrays,
                                           scalars,
                                           node_to_info);
    utils::Hasher hasher;
    hasher.add(optype_hash)
          .add(desired_computation_rank)
          .add(node_to_info->at(dest_.expression().get()).hash)
          .add(node_to_info->at(root_.expression().get()).hash);
    (*node_to_info)[this].hash = hasher.value();
}

std::string JITRunner::assignment_code(const symbol_table_t& symbol_table,
                                       const node_to_info_t& node_to_info,
                                       memory::DeviceT device_type) const {
    std::string dest_call_code = op::jit::get_call_code_nd(dest_, symbol_table, node_to_info, device_type);
    auto root = as_jit_node(root_);
    int computation_rank = node_to_info.at(this).computation_rank;
    std::string indexing_nd = computation_rank == 1 ? "(i)" : "[" + generate_accessor_string(computation_rank) + "]";
    return utils::make_message(
        dest_call_code, indexing_nd, " ",
        operator_to_name(operator_t_),
        " ",
        root->get_call_code_nd(symbol_table, node_to_info, device_type), indexing_nd, ";\n"
    );
}

std::string JITRunner::get_call_code_nd(const symbol_table_t& symbol_table,
                                        const node_to_info_t& node_to_info,
                                        memory::DeviceT device_type) const {
    std::string dest_call_code = op::jit::get_call_code_nd(dest_, symbol_table, node_to_info, device_type);

    auto root = as_jit_node(root_);
    int computation_rank = node_to_info.at(this).computation_rank;
    if (device_type == memory::DEVICE_T_CPU) {
        if (computation_rank == 1) {
            return utils::make_message(
                "    int num_el = ", dest_call_code, ".shape().numel();\n",
                "    #pragma clang loop vectorize(enable)\n",
                "    #pragma clang loop interleave(enable)\n",
                "    for (int i = 0; i < num_el; ++i) {\n",
                "        ", assignment_code(symbol_table, node_to_info, device_type),
                "    }\n"
            );
        } else {
            return construct_for_loop(
                computation_rank,
                assignment_code(symbol_table, node_to_info, device_type),
                dest_call_code,
                4
            );
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
                "        ", root->get_call_code_nd(symbol_table, node_to_info, device_type), ",\n"
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
                "        ", root->get_call_code_nd(symbol_table, node_to_info, device_type), ",\n"
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
                                         const std::vector<const BufferView*>& arrays,
                                         const std::vector<const ScalarView*>& scalars,
                                         const node_to_info_t& node_to_info) const {
    std::unordered_set<hash_t> prefix_code_visited;
    std::stringstream result;
    result << prefix_code(node_to_info, device.type());
    result << as_jit_node(root_)->prefix_code(node_to_info, device.type());
    root_.expression()->for_all_suboperations([&](const Array& arr) {
        if (is_jit_node(arr)) {
            auto pc      = as_jit_node(arr)->prefix_code(node_to_info, device.type());
            auto pc_hash = utils::get_hash(pc);
            if (prefix_code_visited.find(pc_hash) == prefix_code_visited.end()) {
                result << pc;
                prefix_code_visited.insert(pc_hash);
            }
        }
    });
    // for (auto kv : node_to_info) {
    //     std::cout << typeid(*kv.first).name()
    //               << ".computation_rank=" << kv.second.computation_rank
    //               << ", .computation_shape= " << kv.second.computation_shape << std::endl;
    // }

    result << "void run(void** array_data, const int* offsets, const int** sizes, const int** strides, const void** scalar_arguments) {\n";

    // DECLARE SYMBOLS
    symbol_table_t symbol_table;
    for (int i = 0; i < arrays.size(); ++i) {
        auto name = utils::make_message("array_", i, "_view");

        symbol_table[(const Expression*)arrays[i]] = name;
        if (arrays[i]->contiguous_memory()) {
            result << build_array_definition(
                dtype_to_cpp_name(arrays[i]->dtype_),
                name,
                true,
                node_to_info.at((const Expression*)arrays[i]).computation_rank,
                utils::make_message("array_data[", i, "], offsets[", i, "], sizes[", i, "]")
            );
        } else {
            result << build_array_definition(
                dtype_to_cpp_name(arrays[i]->dtype_),
                name,
                false,
                node_to_info.at((const Expression*)arrays[i]).computation_rank,
                utils::make_message("array_data[", i, "], offsets[", i, "], sizes[", i, "], strides[", i, "]")
            );
        }
    }

    for (int i = 0; i < scalars.size(); ++i) {
        auto name = utils::make_message("scalar_", i, "_view");

        symbol_table[(const Expression*)scalars[i]] = name;

        result << build_scalar_definition(
            dtype_to_cpp_name(scalars[i]->dtype_),
            name,
            node_to_info.at(scalars[i]).computation_rank,
            utils::make_message("scalar_arguments[", i, "]")
        );
    }
    result << get_call_code_nd(symbol_table, node_to_info, device.type());
    result << "}\n";
    return result.str();
}


std::string JITRunner::prefix_code(const node_to_info_t& node_to_info,
                                   memory::DeviceT device_type) const {
#ifdef DALI_USE_CUDA
    if (device_type == memory::DEVICE_T_GPU) {
        if (node_to_info.at(this).computation_rank == 1) {
            return utils::make_message(
                "template<typename Destination, typename Source>\n"
                "void __global__\n"
                "assign_kernel(Destination dst, Source src, int num_el) {\n"
                "    int idx = blockDim.x * blockIdx.x + threadIdx.x;\n"
                "    int stride = blockDim.x * gridDim.x;\n"
                "    for (int i = idx; i < num_el; i += stride) {\n"
                "        dst(i) ", operator_to_name(operator_t_), " src(i);\n"
                "    }\n"
                "}\n"
            );
        } else {
            return utils::make_message(
                "template<typename Destination, typename Source>\n"
                "void __global__\n"
                "assign_kernel(Destination dst, Source src, int num_el, Shape<", node_to_info.at(this).computation_rank, "> shape) {\n"
                "    int idx = blockDim.x * blockIdx.x + threadIdx.x;\n"
                "    int stride = blockDim.x * gridDim.x;\n"
                "    for (int i = idx; i < num_el; i += stride) {\n"
                "        auto nd_idx = index_to_dim(idx, shape);\n"
                "        dst[nd_idx] ", operator_to_name(operator_t_), " src[nd_idx];\n"
                "    }\n"
                "}\n"
            );
        }
    }
#endif
    return "";
}

std::function<void(void**, const int*, const int**, const int**, const void**)> JITRunner::compile(
            memory::Device device,
            const std::vector<const BufferView*>& arrays,
            const std::vector<const ScalarView*>& scalars,
            const node_to_info_t& node_to_info) const {
    DALI_SCOPE("get_function");
    // compute a quasi-unique hash for the fused operation
    hash_t hash = utils::Hasher().add((int)device.type())
                                 .add(node_to_info.at(this).hash)
                                 .value();
    // check if the operation needs to be runtime compiled
    if (!array_op_compiler.load(hash) || should_always_recompile()) {
        DALI_SCOPE("compilation");
        auto code_template = get_code_template(
            device,
            arrays,
            scalars,
            node_to_info
        );
        array_op_compiler.compile<void**, const int*, const int**, const int**, const void**>(
            hash,
            code_template,
            device.type()
        );
    }
    // return the operation that was loaded or compiled:
    return array_op_compiler.get_function<void**, const int*, const int**, const int**, const void**>(hash);
}

Array jit_root(const Array& array) {
    if (is_jit_runner(array)) {
        return as_jit_runner(array)->root_;
    }
    return array;
}

std::tuple<Array, Array> replace_assign_with_inplace(const Array& node) {
    auto assign = as_assignment(node);
    auto rightside = jit_root(assign->right_);
    auto operator_t = assign->operator_t_;
    if (operator_t == OPERATOR_T_EQL) {
        return std::tuple<Array, Array>(rightside, Array());
    } else if (operator_t == OPERATOR_T_ADD) {
        return std::tuple<Array, Array>(op::add(assign->left_, rightside), assign->left_);
    } else if (operator_t == OPERATOR_T_SUB) {
        return std::tuple<Array, Array>(op::subtract(assign->left_, rightside), assign->left_);
    } else if (operator_t == OPERATOR_T_MUL) {
        return std::tuple<Array, Array>(op::eltmul(assign->left_, rightside), assign->left_);
    } else if (operator_t == OPERATOR_T_DIV) {
        return std::tuple<Array, Array>(op::eltdiv(assign->left_, rightside), assign->left_);
    } else {
        throw std::runtime_error(utils::make_message("No way to replace_assign_with_inplace using operator ",
                                                     operator_to_name(operator_t), "."));
    }
}

Array jit_merge(const Array& root) {
    std::vector<Array> leaves;
    auto assign = as_assignment(root);
    auto root_buffer = assign->left_;
    auto root_operator = assign->operator_t_;
    Array left_leaf, replaced;
    for (auto& arg : right_args(root)) {
        if (arg.is_assignment() &&
            is_jit_runner(as_assignment(arg)->right_)) {
            // grab leaves from existing jit-runner recursively:
            auto extra_leaves = as_jit_runner(as_assignment(arg)->right_)->leaves_;
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
        } else {
            // this node is either an assignment, or a buffer,
            // and is needed as an input here:
            leaves.emplace_back(arg);
        }
    }

    auto new_root = assign->right_;
    return Array(std::make_shared<Assignment>(
        // keep the original target buffer:
        root_buffer, root_operator,
        // use the merged operation instead
        Array(std::make_shared<JITRunner>(new_root, leaves, root_operator, root_buffer))));
}

// JIT RUNNER-IMPL //
struct JITRunnerImpl : public Computation {
    using Computation::Computation;
    void run() {
        auto jit_left = as_jit_node(left_);
        auto jit_right = as_jit_runner(right_);
        int desired_computation_rank = std::max(
            jit_left->min_computation_rank_,
            jit_right->min_computation_rank_
        );
        std::vector<const BufferView*> array_ops;
        std::vector<const ScalarView*> scalar_ops;
        node_to_info_t node_to_info;

        jit_right->compute_node_compilation_info(desired_computation_rank,
                                                 left_.shape(),
                                                 &array_ops,
                                                 &scalar_ops,
                                                 &node_to_info);

        auto device = jit_right->preferred_device();
        auto compiled_self = jit_right->compile(device,
                                                array_ops,
                                                scalar_ops,
                                                node_to_info);
        std::vector<Array> arrays;
        std::transform(array_ops.begin(),
                       array_ops.end(),
                       std::back_inserter(arrays),
                       [&node_to_info](const BufferView* op) {
                           const auto& rank  = node_to_info.at(op).computation_rank;
                           const auto& shape = node_to_info.at(op).computation_shape;
                           if (rank == op->ndim()) {
                               return op->reshape_broadcasted(shape);
                           } else if (rank == 1) {
                               return op->reshape_broadcasted(shape)->copyless_ravel();
                           } else {
                               return op->reshape_broadcasted(shape)->copyless_right_fit_ndim(rank);
                           }
                       });
        std::vector<const void*> scalars;
        std::transform(scalar_ops.begin(),
                       scalar_ops.end(),
                       std::back_inserter(scalars),
                       [&](const ScalarView* op) {
                            return op->value_ptr();
                       });

        std::vector<void*> data_ptrs;
        std::vector<int> offsets;
        std::vector<const int*> shapes;
        std::vector<const int*> strides;
        for (auto& arr : arrays) {
            data_ptrs.push_back(arr.memory()->mutable_data(device));
            offsets.push_back(arr.offset());
            shapes.push_back(arr.shape().data());
            strides.push_back(arr.strides().data());
        }
        // std::string assign_name;
        // if (Scope::has_observers()) {
        //     assign_name = ExpressionState::full_operation_name();
        // }
        // DALI_SCOPE(assign_name);
        // std::cout << "running " << assign_name << std::endl;
        compiled_self(
            data_ptrs.data(),
            offsets.data(),
            shapes.data(),
            strides.data(),
            scalars.data()
        );
    }
};


Array buffer_buffer_op(Array node) {
    auto assignment = std::dynamic_pointer_cast<Assignment>(node.expression());

    // TODO(jonathan): this should not be needed
    auto identity_node = op::identity(assignment->right_);
    auto something = std::make_shared<op::jit::JITRunner>(
        identity_node,
        std::vector<Array>({assignment->right_}),
        OPERATOR_T_EQL,
        assignment->left_
    );
    return Array(
        std::make_shared<Assignment>(
            assignment->left_,
            assignment->operator_t_,
            Array(something)
        )
    );
}

int min_computation_rank(const Array& array) {
    if (array.is_buffer() || array.is_assignment() || array.is_control_flow()) {
        return array.strides().empty() ? 1 : array.ndim();
    } else {
        return as_jit_node(array)->min_computation_rank_;
    }
}

void compute_node_compilation_info(const Array& a,
                                   int desired_computation_rank,
                                   const std::vector<int>& desired_computation_shape,
                                   std::vector<const BufferView*>* arrays,
                                   std::vector<const ScalarView*>* scalars,
                                   node_to_info_t* node_to_info) {
    if (a.is_buffer()) {
        buffer_compute_node_compilation_info(a,
                                             desired_computation_rank,
                                             desired_computation_shape,
                                             arrays,
                                             scalars,
                                             node_to_info);
    } else if (is_jit_node(a)) {
        as_jit_node(a)->compute_node_compilation_info(
            desired_computation_rank,
            desired_computation_shape,
            arrays,
            scalars,
            node_to_info);
    } else {
        throw std::runtime_error(utils::make_message(
            "Can only compute node compilation info for JITNode "
            "or BufferView (got ", a.expression_name(), ")."));
    }
}


std::string buffer_get_call_code_nd(const Array& array,
                                    const symbol_table_t& symbol_table,
                                    const node_to_info_t& node_to_info,
                                    memory::DeviceT device_type) {
    return symbol_table.at(array.expression().get());
}


std::string get_call_code_nd(const Array& a,
                             const symbol_table_t& symbol_table,
                             const node_to_info_t& node_to_info,
                             memory::DeviceT device_type) {
    if (a.is_buffer()) {
        return buffer_get_call_code_nd(a, symbol_table, node_to_info, device_type);
    } else if (is_jit_node(a)) {
        return as_jit_node(a)->get_call_code_nd(symbol_table, node_to_info, device_type);
    } else {
        throw std::runtime_error(utils::make_message(
            "Can only create call code for JITNode "
            "or BufferView (got ", a.expression_name(), ")."));
    }
}

int registered_opt = register_optimization(is_jit_assignment, jit_merge);
int registered_impl = register_implementation(
   typeid(JITRunner).name(),
   [](Array dest, OPERATOR_T operator_t, Array x) -> std::shared_ptr<Computation> {
        return std::make_shared<JITRunnerImpl>(dest, operator_t, x);
   });
}  // namespace jit
}  // namespace op
