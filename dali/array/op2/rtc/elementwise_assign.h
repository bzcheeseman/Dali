#ifndef DALI_ARRAY_OP_RTC_ELEMENTWISE_ASSIGN_H
#define DALI_ARRAY_OP_RTC_ELEMENTWISE_ASSIGN_H

/*
struct ElementwiseAssignExpressionState : public AbstractAssignExpressionState, Runnable {
    static const hash_t optype_hash;

    using AbstractAssignExpressionState::AbstractAssignExpressionState;

    // that line that Szymon deleted:
    // ExpressionState(std::max(left->min_computation_rank_, right->min_computation_rank_)),


    virtual std::string name() const {
        return "elementwise_assign";
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





    virtual void compute_node_compilation_info(int desired_computation_rank,
                                               const std::vector<int>& desired_computation_shape,
                                               std::vector<const ArrayWrapper*>* arrays,
                                               std::vector<const ScalarWrapper*>* scalars,
                                               node_to_info_t* node_to_info) const {
        (*node_to_info)[this].computation_rank  = desired_computation_rank;
        (*node_to_info)[this].computation_shape = desired_computation_shape;
        left_->compute_node_compilation_info(desired_computation_rank, desired_computation_shape, arrays, scalars, node_to_info);
        right_->compute_node_compilation_info(desired_computation_rank, desired_computation_shape, arrays, scalars, node_to_info);
        (*node_to_info)[this].hash = Hasher().add(operator_t_)
                                             .add(optype_hash)
                                             .add(node_to_info->at(left_).hash)
                                             .add(node_to_info->at(right_).hash)
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
};
const hash_t ElementwiseAssignExpressionState::optype_hash = std::hash<std::string>()("ElementwiseAssignExpressionState");
*/

#endif  // DALI_ARRAY_OP_RTC_ELEMENTWISE_ASSIGN_H
