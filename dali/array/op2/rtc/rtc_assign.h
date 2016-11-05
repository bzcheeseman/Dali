#ifndef DALI_ARRAY_OP_RTC_RTC_ASSIGN_H
#define DALI_ARRAY_OP_RTC_RTC_ASSIGN_H

#include "dali/array/op2/rtc_utils.h"

namespace expression {
namespace rtc {

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

struct RtcAssignExpressionState : virtual public Runnable {
    typedef RtcExpression::node_to_info_t node_to_info_t;
    typedef RtcExpression::symbol_table_t symbol_table_t;
    static const hash_t optype_hash;

    std::shared_ptr<const RtcExpression> left_;
    std::shared_ptr<const RtcExpression> right_;
    OPERATOR_T operator_t_;
    memory::Device device_;

    RtcAssignExpressionState(std::shared_ptr<const RtcExpression> left,
                             OPERATOR_T operator_t,
                             std::shared_ptr<const RtcExpression> right,
                             memory::Device device) :
        left_(left), right_(right), operator_t_(operator_t), device_(device) {}

    virtual std::string name() const {
        return "elementwise_assign";
    }

    std::vector<std::shared_ptr<const ExpressionState>> arguments() const {
        return {left_, right_};
    }

    virtual std::string prefix_code(
            const node_to_info_t& node_to_info,
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

    virtual DType dtype() const {
        return left_->dtype();
    }

    virtual int ndim() const {
        return left_->ndim();
    }

    virtual std::vector<int> bshape() const {
        return left_->bshape();
    }

    virtual std::shared_ptr<const ExpressionState> destination_op() const {
        return left_;
    }

    virtual void compute_node_compilation_info(int desired_computation_rank,
                                               const std::vector<int>& desired_computation_shape,
                                               std::vector<const RtcArrayWrapper*>* arrays,
                                               std::vector<const ScalarWrapper*>* scalars,
                                               node_to_info_t* node_to_info) const {
        (*node_to_info)[this].computation_rank  = desired_computation_rank;
        (*node_to_info)[this].computation_shape = desired_computation_shape;
        left_->compute_node_compilation_info(desired_computation_rank, desired_computation_shape, arrays, scalars, node_to_info);
        right_->compute_node_compilation_info(desired_computation_rank, desired_computation_shape, arrays, scalars, node_to_info);
        (*node_to_info)[this].hash = utils::Hasher().add(operator_t_)
                                                    .add(optype_hash)
                                                    .add(node_to_info->at(left_.get()).hash)
                                                    .add(node_to_info->at(right_.get()).hash)
                                                    .add(desired_computation_rank).value();
    }


    virtual std::string assignment_code(const symbol_table_t& symbol_table,
                                        const node_to_info_t& node_to_info,
                                        memory::DeviceT device_type) const {
        int computation_rank = node_to_info.at(this).computation_rank;
        std::string indexing_nd = computation_rank == 1 ? "(i)" : "[" + generate_accessor_string(computation_rank) + "]";
        return utils::make_message(
            left_->get_call_code_nd(symbol_table, node_to_info, device_type), indexing_nd, " ",
            operator_to_name(operator_t_),
            " ",
            right_->get_call_code_nd(symbol_table, node_to_info, device_type), indexing_nd, ";\n"
        );
    }

    virtual std::string get_call_code_nd(const symbol_table_t& symbol_table,
                                         const node_to_info_t& node_to_info,
                                         memory::DeviceT device_type) const {
        int computation_rank = node_to_info.at(this).computation_rank;
        if (device_type == memory::DEVICE_T_CPU) {
            // TODO: debate if we want to allow chaining here:
            //       (e.g. call this assignment, then this assignment, etc...)
            if (computation_rank == 1) {
                return utils::make_message(
                    "    int num_el = ", left_->get_call_code_nd(symbol_table, node_to_info, device_type), ".shape().numel();\n",
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
                    left_->get_call_code_nd(symbol_table, node_to_info, device_type),
                    4
                );
            }
        }
#ifdef DALI_USE_CUDA
        else if (device_type == memory::DEVICE_T_GPU) {
            if (computation_rank == 1) {
                return utils::make_message(
                        "    int num_el = ", left_->get_call_code_nd(symbol_table, node_to_info, device_type), ".shape().numel();\n"
                        "    const int NT = 128;\n"
                        "    // const int MAX_BLOCKS = 40960;\n"
                        "    int grid_size = div_ceil(num_el, NT);\n"
                        "    // assert(grid_size <= MAX_BLOCKS);\n"
                        "    assign_kernel<<<grid_size, NT, 0, NULL>>>(\n"
                        "        ", left_->get_call_code_nd(symbol_table, node_to_info, device_type), ",\n"
                        "        ", right_->get_call_code_nd(symbol_table, node_to_info, device_type), ",\n"
                        "        num_el\n"
                        "    );\n"
                );
            } else {
                return utils::make_message(
                        "    auto shape = ", left_->get_call_code_nd(symbol_table, node_to_info, device_type), ".shape();\n"
                        "    int num_el = shape.numel();\n"
                        "    const int NT = 128;\n"
                        "    // const int MAX_BLOCKS = 40960;\n"
                        "    int grid_size = div_ceil(num_el, NT);\n"
                        "    // assert(grid_size <= MAX_BLOCKS);\n"
                        "    assign_kernel<<<grid_size, NT, 0, NULL>>>(\n"
                        "        ", left_->get_call_code_nd(symbol_table, node_to_info, device_type), ",\n"
                        "        ", right_->get_call_code_nd(symbol_table, node_to_info, device_type), ",\n"
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

    virtual void run() const {
        int desired_computation_rank = std::max(
            left_->min_computation_rank_,
            right_->min_computation_rank_
        );
        std::vector<const RtcArrayWrapper*> array_ops;
        std::vector<const ScalarWrapper*> scalar_ops;
        node_to_info_t node_to_info;

        compute_node_compilation_info(desired_computation_rank,
                                      left_->shape(),
                                      &array_ops,
                                      &scalar_ops,
                                      &node_to_info);

        auto compiled_self = compile(array_ops,
                                     scalar_ops,
                                     node_to_info);
        std::vector<Array> arrays;
        std::transform(array_ops.begin(),
                       array_ops.end(),
                       std::back_inserter(arrays),
                       [&node_to_info](const RtcArrayWrapper* op) {
                           const auto& rank  = node_to_info.at(op).computation_rank;
                           const auto& shape = node_to_info.at(op).computation_shape;
                           if (rank == op->ndim()) {
                               return op->array_.reshape_broadcasted(shape);
                           } else if (rank == 1) {
                               return op->array_.reshape_broadcasted(shape).copyless_ravel();
                           } else {
                               return op->array_.reshape_broadcasted(shape).copyless_right_fit_ndim(rank);
                           }
                       });
        std::vector<const void*> scalars;
        std::transform(scalar_ops.begin(),
                       scalar_ops.end(),
                       std::back_inserter(scalars),
                       [&](const ScalarWrapper* op) {
                            return op->value_ptr();
                       });

        std::vector<void*> data_ptrs;
        std::vector<int> offsets;
        std::vector<const int*> shapes;
        std::vector<const int*> strides;
        for (auto& arr : arrays) {
            data_ptrs.push_back(arr.memory()->mutable_data(device_));
            offsets.push_back(arr.offset());
            shapes.push_back(arr.shape().data());
            strides.push_back(arr.strides().data());
        }

        std::string assign_name;
        if (Scope::has_observers()) {
            assign_name = full_operation_name();
        }
        DALI_SCOPE(assign_name);
        compiled_self(
            data_ptrs.data(),
            offsets.data(),
            shapes.data(),
            strides.data(),
            scalars.data()
        );
    }

    std::function<void(void**, const int*, const int**, const int**, const void**)> compile(
                const std::vector<const RtcArrayWrapper*>& arrays,
                const std::vector<const ScalarWrapper*>& scalars,
                const node_to_info_t& node_to_info) const {
        DALI_SCOPE("get_function");
        // compute a quasi-unique hash for the fused operation
        hash_t hash = utils::Hasher().add((int)device_.type())
                                     .add(node_to_info.at(this).hash)
                                     .value();
        // check if the operation needs to be runtime compiled
        if (!array_op_compiler.load(hash) || should_always_recompile()) {
            DALI_SCOPE("compilation");
            auto code_template = get_code_template(
                device_,
                arrays,
                scalars,
                node_to_info
            );
            array_op_compiler.compile<void**, const int*, const int**, const int**, const void**>(
                hash,
                code_template,
                device_.type()
            );
        }
        // return the operation that was loaded or compiled:
        return array_op_compiler.get_function<void**, const int*, const int**, const int**, const void**>(hash);
    }

    virtual std::string get_code_template(memory::Device device,
                                          const std::vector<const RtcArrayWrapper*>& arrays,
                                          const std::vector<const ScalarWrapper*>& scalars,
                                          const node_to_info_t& node_to_info) const final {
        std::unordered_set<hash_t> prefix_code_visited;
        std::stringstream result;

        this->for_all_suboperations([&](const ExpressionState* node) {
            auto jit_node = node->as_jit();
            if (jit_node) {
                auto pc      = jit_node->prefix_code(node_to_info, device.type());
                auto pc_hash = utils::get_hash(pc);
                if (prefix_code_visited.find(pc_hash) == prefix_code_visited.end()) {
                    result << pc;
                    prefix_code_visited.insert(pc_hash);
                }
            }
        });

        result << "void run(void** array_data, const int* offsets, const int** sizes, const int** strides, const void** scalar_arguments) {\n";

        // DECLARE SYMBOLS
        symbol_table_t symbol_table;
        for (int i = 0; i < arrays.size(); ++i) {
            auto name = utils::make_message("array_", i, "_view");

            symbol_table[(const RtcExpression*)arrays[i]] = name;
            if (arrays[i]->contiguous()) {
                result << build_array_definition(
                    dtype_to_cpp_name(arrays[i]->dtype()),
                    name,
                    true,
                    node_to_info.at((const RtcExpression*)arrays[i]).computation_rank,
                    utils::make_message("array_data[", i, "], offsets[", i, "], sizes[", i, "]")
                );
            } else {
                result << build_array_definition(
                    dtype_to_cpp_name(arrays[i]->dtype()),
                    name,
                    false,
                    node_to_info.at((const RtcExpression*)arrays[i]).computation_rank,
                    utils::make_message("array_data[", i, "], offsets[", i, "], sizes[", i, "], strides[", i, "]")
                );
            }
        }

        for (int i = 0; i < scalars.size(); ++i) {
            auto name = utils::make_message("scalar_", i, "_view");

            symbol_table[(const RtcExpression*)scalars[i]] = name;

            result << build_scalar_definition(
                dtype_to_cpp_name(scalars[i]->dtype()),
                name,
                node_to_info.at(scalars[i]).computation_rank,
                utils::make_message("scalar_arguments[", i, "]")
            );
        }
        result << get_call_code_nd(symbol_table, node_to_info, device.type());
        result << "}\n";
        return result.str();
    }
};
const hash_t RtcAssignExpressionState::optype_hash = std::hash<std::string>()(
    "RtcAssignExpressionState"
);

}  // namespace rtc
}  // namespace expression


#endif  // DALI_ARRAY_OP_RTC_RTC_ASSIGN_H
