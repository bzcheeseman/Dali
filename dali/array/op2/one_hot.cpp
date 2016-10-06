#include "one_hot.h"

#include "dali/array/op2/operation.h"
#include "dali/array/op2/elementwise_operation.h"
#include "dali/array/op2/rtc_utils.h"
#include "dali/utils/hash_utils.h"
#include "dali/utils/make_message.h"

struct OneHotOperationState : public OperationState {
    static const hash_t optype_hash;

    operation_state_ptr indices_;
    operation_state_ptr on_value_;
    operation_state_ptr off_value_;
    int depth_;
    operation_state_ptr depth_operation_;

    OneHotOperationState(operation_state_ptr indices,
                         int depth,
                         operation_state_ptr on_value,
                         operation_state_ptr off_value) :
            OperationState(indices->min_computation_rank_ + 1),
            indices_(indices),
            on_value_(on_value),
            off_value_(off_value),
            depth_(depth),
            depth_operation_(Operation(depth).state_) {
    }

    std::string prefix_code(const node_to_info_t& node_to_info) const {
        return"template<typename C1, typename C2, typename C3, typename C4>\n"
        "struct OneHotKernel {\n"
        "    const C1& indices_;\n"
        "    const C2& depth_;\n"
        "    const C3& on_value_;\n"
        "    const C4& off_value_;\n"
        "    static const int ndim = C1::ndim + 1;\n"
        "    typedef typename C3::T T;\n"
        "    XINLINE OneHotKernel(const C1& indices,\n"
        "                         const C2& depth,\n"
        "                         const C3& on_value,\n"
        "                         const C4& off_value)\n"
        "        : indices_(indices), depth_(depth), on_value_(on_value),\n"
        "          off_value_(off_value) {}\n"
        "    XINLINE T operator[](Shape<ndim> query) {\n"
        "        if (indices_[query.axis_reduced_shape()] == query[ndim - 1]) {\n"
        "             return on_value_(0);\n"
        "        } else {\n"
        "             return off_value_(0);\n"
        "        }\n"
        "    }\n"
        "};\n"
        "template<typename C1, typename C2, typename C3, typename C4>\n"
        "OneHotKernel<C1, C2, C3, C4> one_hot_kernel(const C1& a, const C2& b, const C3& c, const C4& d) {\n"
        "    return OneHotKernel<C1, C2, C3, C4>(a, b, c, d);\n"
        "}\n";
    }

    DType dtype() const {
        return on_value_->dtype();
    }

    std::vector<int> bshape() const {
        auto result = indices_->bshape();
        result.emplace_back(depth_);
        return result;
    }

    std::vector<operation_state_ptr> arguments() const {
        return {indices_, on_value_, off_value_, depth_operation_};
    }

    bool is_dim_collapsible_with_dim_minus_one(const int& dim) const {
        if (dim == ndim() - 1) {
            return false;
        }
        return indices_->is_dim_collapsible_with_dim_minus_one(dim);
    }

    operation_state_ptr collapse_dim_with_dim_minus_one(const int& dim) const {
        return std::make_shared<OneHotOperationState>(
            indices_->collapse_dim_with_dim_minus_one(dim),
            depth_, on_value_, off_value_
        );
    }

    operation_state_ptr transpose(const std::vector<int>& permutation) const {
        bool last_dim_unchanged = permutation.back() == int(permutation.size()) - 1;
        if (last_dim_unchanged)
            return std::make_shared<OneHotOperationState>(
                indices_->transpose(permutation),
                depth_,
                on_value_,
                off_value_
            );
        throw std::runtime_error(
            "Cannot transpose last dimension result of one_hot"
        );
        return shared_from_this();
    }

    void compute_node_compilation_info(
            int desired_computation_rank,
            const std::vector<int>& desired_computation_shape,
            std::vector<const ArrayOperationState*>* arrays,
            std::vector<const ScalarOperationState*>* scalars,
            node_to_info_t* node_to_info) const {
        (*node_to_info)[this].computation_rank = desired_computation_rank;
        auto indices_shape = desired_computation_shape;
        indices_shape.pop_back();
        indices_->compute_node_compilation_info(desired_computation_rank - 1, indices_shape, arrays, scalars, node_to_info);
        on_value_->compute_node_compilation_info(1, {}, arrays, scalars, node_to_info);
        off_value_->compute_node_compilation_info(1, {}, arrays, scalars, node_to_info);
        depth_operation_->compute_node_compilation_info(1, {}, arrays, scalars, node_to_info);
        (*node_to_info)[this].hash = utils::Hasher().add(optype_hash)
                                                    .add(desired_computation_rank)
                                                    .add(node_to_info->at(indices_.get()).hash)
                                                    .add(node_to_info->at(on_value_.get()).hash)
                                                    .add(node_to_info->at(off_value_.get()).hash)
                                                    .value();
    }


    std::string get_call_code_nd(
            const symbol_table_t& symbol_table,
            const node_to_info_t& node_to_info) const {
        return utils::make_message("one_hot_kernel(",
                                    indices_->get_call_code_nd(symbol_table, node_to_info),
                                    ",",
                                    depth_operation_->get_call_code_nd(symbol_table, node_to_info),
                                    ",",
                                    on_value_->get_call_code_nd(symbol_table, node_to_info),
                                    ",",
                                    off_value_->get_call_code_nd(symbol_table, node_to_info),
                                    ")");
    }
};

const hash_t OneHotOperationState::optype_hash = std::hash<std::string>()("OneHotOperationState");

namespace op2 {
    Operation one_hot(
            const Operation& indices,
            int depth,
            const Operation& on_value,
            const Operation& off_value) {
        ASSERT2(
            indices.dtype() == DTYPE_INT32,
            utils::make_message("indices must be integers (got ", indices.dtype(), ")")
        );
        ASSERT2(
            on_value.is_scalar(),
            utils::make_message("on_value must be a scalar (got on_value.ndim=", on_value.ndim(), ")")
        );
        ASSERT2(
            off_value.is_scalar(),
            utils::make_message("off_value must be a scalar (got off_value.ndim=", off_value.ndim(), ")")
        );
        auto on_off = ensure_arguments_compatible(on_value, off_value);
        return Operation(std::make_shared<OneHotOperationState>(
            indices.state_, depth, std::get<0>(on_off).state_, std::get<1>(on_off).state_
        ));
    }
}  // namespace op2
