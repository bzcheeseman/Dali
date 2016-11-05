#include "rtc_expression.h"

#include "dali/utils/make_message.h"
#include "dali/array/op2/expression/array_wrapper.h"
#include "dali/array/op2/rtc/scalar_wrapper.h"
#include "dali/array/op2/rtc_utils.h"
#include "dali/array/function2/compiler.h"

using utils::Hasher;

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


namespace expression {
namespace rtc {


    // TODO(jonathan, szymon): move this into member function
    // void eval_op(const Expression& op,
    //              const std::vector<int>& output_shape,
    //              const memory::Device& output_device) {

    //     auto& self = *op.state_;
    //     int desired_computation_rank = self.min_computation_rank_;
    //     std::vector<const ArrayWrapper*> array_ops;
    //     std::vector<const ScalarWrapper*> scalar_ops;
    //     RtcExpression::node_to_info_t node_to_info;

    //     self.compute_node_compilation_info(desired_computation_rank,
    //                                        output_shape,
    //                                        &array_ops,
    //                                        &scalar_ops,
    //                                        &node_to_info);

    //     auto compiled_self = self.compile(output_device,
    //                                       array_ops,
    //                                       scalar_ops,
    //                                       node_to_info);
    //     std::vector<Array> arrays;
    //     std::transform(array_ops.begin(),
    //                    array_ops.end(),
    //                    std::back_inserter(arrays),
    //                    [&node_to_info](const ArrayWrapper* op) {
    //                        const auto& rank  = node_to_info.at(op).computation_rank;
    //                        const auto& shape = node_to_info.at(op).computation_shape;
    //                        if (rank == op->ndim()) {
    //                            return op->array_.reshape_broadcasted(shape);
    //                        } else if (rank == 1) {
    //                            return op->array_.reshape_broadcasted(shape).copyless_ravel();
    //                        } else {
    //                            return op->array_.reshape_broadcasted(shape).copyless_right_fit_ndim(rank);
    //                        }
    //                    });
    //     std::vector<const void*> scalars;
    //     std::transform(scalar_ops.begin(),
    //                    scalar_ops.end(),
    //                    std::back_inserter(scalars),
    //                    [&](const ScalarWrapper* op) {
    //                         return op->value_ptr();
    //                    });

    //     std::vector<void*> data_ptrs;
    //     std::vector<int> offsets;
    //     std::vector<const int*> shapes;
    //     std::vector<const int*> strides;
    //     for (auto& arr : arrays) {
    //         data_ptrs.push_back(arr.memory()->mutable_data(output_device));
    //         offsets.push_back(arr.offset());
    //         shapes.push_back(arr.shape().data());
    //         strides.push_back(arr.strides().data());
    //     }

    //     std::string name;
    //     if (Scope::has_observers()) {
    //         name = op.name();
    //     }
    //     DALI_SCOPE(name);
    //     compiled_self(data_ptrs.data(), offsets.data(), shapes.data(), strides.data(), scalars.data());
    // }




    RtcExpression::RtcExpression(int min_computation_rank) :
            min_computation_rank_(min_computation_rank) {
    }


    std::string RtcExpression::prefix_code(const node_to_info_t& node_to_info, memory::DeviceT device_type) const {
        return "";
    }


    std::string RtcExpression::get_code_template(memory::Device device,
                                                 const std::vector<const expression::ArrayWrapper*>& arrays,
                                                 const std::vector<const ScalarWrapper*>& scalars,
                                                 const node_to_info_t& node_to_info) const {
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


    std::function<void(void**, const int*, const int**, const int**, const void**)> RtcExpression::compile(
            memory::Device device,
            const std::vector<const expression::ArrayWrapper*>& arrays,
            const std::vector<const ScalarWrapper*>& scalars,
            const node_to_info_t& node_to_info) const {
        DALI_SCOPE("get_function");
        // compute a quasi-unique hash for the fused operation
        hash_t hash = Hasher().add((int)device.type())
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


    bool RtcExpression::is_dim_collapsible_with_dim_minus_one(const int& dim) const {
        return false;
    }

    std::shared_ptr<const RtcExpression> RtcExpression::collapse_dim_with_dim_minus_one(const int& dim) const {
        return jit_shared_from_this();
    }

    std::shared_ptr<const RtcExpression> RtcExpression::transpose(const std::vector<int>& permutation) const {
        ASSERT2(false, "Transpose not implemented for this Expression.");
        return jit_shared_from_this();
    }

    bool RtcExpression::is_assignable() const {
        return false;
    }

    std::shared_ptr<const RtcExpression> RtcExpression::jit_shared_from_this() const {
        return std::dynamic_pointer_cast<const RtcExpression>(shared_from_this());
    }

    std::shared_ptr<RtcExpression> RtcExpression::jit_shared_from_this() {
        return std::dynamic_pointer_cast<RtcExpression>(shared_from_this());
    }

}  // namespace rtc
}  // namespace expression
