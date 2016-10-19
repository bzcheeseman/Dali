#include "outer.h"

#include "dali/array/op2/operation.h"
#include "dali/array/op2/elementwise_operation.h"
#include "dali/array/op2/rtc_utils.h"
#include "dali/utils/hash_utils.h"
#include "dali/utils/make_message.h"

struct OuterOperationState : public OperationState {
    static const hash_t optype_hash;
    operation_state_ptr left_;
    operation_state_ptr right_;

    OuterOperationState(operation_state_ptr left, operation_state_ptr right)
        : OperationState(2), left_(left), right_(right) {}

    virtual std::string name() const {
        return "outer";
    }

    std::string prefix_code(const node_to_info_t& node_to_info, memory::DeviceT device_type) const {
        return"template<typename C1, typename C2>\n"
        "struct OuterKernel {\n"
        "    const C1& left_;\n"
        "    const C2& right_;\n"
        "    static const int ndim = 2;\n"
        "    typedef typename C1::T T;\n"
        "    XINLINE OuterKernel(const C1& left, const C2& right)\n"
        "        : left_(left), right_(right) {}\n"
        "    XINLINE T operator[](Shape<ndim> query) {\n"
        "        return left_(query[ndim - 2]) * right_(query[ndim - 1]);\n"
        "    }\n"
        "};\n"
        "template<typename C1, typename C2>\n"
        "OuterKernel<C1, C2> outer_kernel(const C1& a, const C2& b) {\n"
        "    return OuterKernel<C1, C2>(a, b);\n"
        "}\n";
    }

    DType dtype() const {
        return left_->dtype();
    }

    std::vector<int> bshape() const {
        std::vector<int> result = {1, 1};
        auto left_bshape = left_->bshape();
        if (left_bshape.size() > 0) {result[0] = left_bshape[0];}
        auto right_bshape = right_->bshape();
        if (right_bshape.size() > 0) {result[1] = right_bshape[0];}
        return result;
    }

    int ndim() const {
        return 2;
    }

    std::vector<operation_state_ptr> arguments() const {
        return {left_, right_};
    }

    bool is_dim_collapsible_with_dim_minus_one(const int& dim) const {
        return false;
    }

    operation_state_ptr collapse_dim_with_dim_minus_one(const int& dim) const {
        throw std::runtime_error("cannot collapse dim with dim minus one the result of outer.");
        return shared_from_this();
    }

    operation_state_ptr transpose(const std::vector<int>& permutation) const {
        ASSERT2(permutation.size() == 2, utils::make_message("transpose of "
            "outer must receive 2 axes (got ", permutation, ")."));
        // 1) No change:
        if (permutation[0] == 0 && permutation[1] == 1) {
            return shared_from_this();
        }
        // 2) transpose of outer is equivalent to swapping order of arguments:
        return std::make_shared<OuterOperationState>(right_, left_);
    }

    void compute_node_compilation_info(
            int desired_computation_rank,
            const std::vector<int>& desired_computation_shape,
            std::vector<const ArrayOperationState*>* arrays,
            std::vector<const ScalarOperationState*>* scalars,
            node_to_info_t* node_to_info) const {
        (*node_to_info)[this].computation_rank = desired_computation_rank;
        left_->compute_node_compilation_info(
            1,
            {desired_computation_shape[desired_computation_shape.size() - 2]},
            arrays,
            scalars,
            node_to_info
        );
        right_->compute_node_compilation_info(
            1,
            {desired_computation_shape[desired_computation_shape.size() - 1]},
            arrays,
            scalars,
            node_to_info
        );
        (*node_to_info)[this].hash = utils::Hasher().add(optype_hash)
                                                    .add(desired_computation_rank)
                                                    .add(node_to_info->at(left_.get()).hash)
                                                    .add(node_to_info->at(right_.get()).hash)
                                                    .value();
    }


    std::string get_call_code_nd(
            const symbol_table_t& symbol_table,
            const node_to_info_t& node_to_info,
            memory::DeviceT device_type) const {
        return utils::make_message("outer_kernel(",
                                    left_->get_call_code_nd(symbol_table, node_to_info, device_type),
                                    ",",
                                    right_->get_call_code_nd(symbol_table, node_to_info, device_type),
                                    ")");
    }
};

const hash_t OuterOperationState::optype_hash = std::hash<std::string>()("OuterOperationState");

namespace op {
    Operation outer(const Operation& left, const Operation& right) {
        ASSERT2(left.ndim() <= 1, utils::make_message("left operand of outer "
            "must have ndim <= 1 (got ", left.ndim(), ")."));
        ASSERT2(right.ndim() <= 1, utils::make_message("right operand of outer "
            "must have ndim <= 1 (got ", right.ndim(), ")."));
        auto left_right = ensure_arguments_compatible(left, right);
        return Operation(std::make_shared<OuterOperationState>(
            std::get<0>(left_right).state_, std::get<1>(left_right).state_
        ));
    }
}  // namespace op2
