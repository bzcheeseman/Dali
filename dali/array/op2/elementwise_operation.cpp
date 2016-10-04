#include "elementwise_operation.h"

#include <vector>
#include "dali/utils/hash_utils.h"
#include "dali/utils/assert2.h"
#include "dali/array/op2/operation.h"
#include "dali/array/op2/rtc_utils.h"
#include "dali/array/op2/elementwise_kernel_utils.h"

DType type_promotion(const Operation& a, const Operation& b) {
    // TODO(jonathan,szymon) speed up this function
    bool a_scalar = a.is_scalar();
    bool b_scalar = b.is_scalar();

    if (a_scalar ^ b_scalar == 0) {
        // if they are both scalars or both arrays
        if (a.dtype() == DTYPE_DOUBLE || b.dtype() == DTYPE_DOUBLE) {
            return DTYPE_DOUBLE;
        } else if (a.dtype() == DTYPE_FLOAT || b.dtype() == DTYPE_FLOAT) {
            return DTYPE_FLOAT;
        } else {
            return DTYPE_INT32;
        }
    } else if (a_scalar) {
        // if a is scalar and b is array.
        return b.dtype();
    } else {
        // if a is array and b is scalar.
        return a.dtype();
    }
}

bool ndim_compatible(const Operation& a, const Operation& b) {
    int a_ndim = a.ndim();
    int b_ndim = b.ndim();
    return a_ndim == 0 || b_ndim == 0 || a_ndim == b_ndim;
}


///////////////////////////////////////////////////////////////////////////////
//                       HEADERS                                             //
///////////////////////////////////////////////////////////////////////////////

struct ElementwiseOperationState : public OperationState {
    static const hash_t optype_hash;

    const operation_state_ptrs arguments_;
    const std::string functor_name_;

    static int compute_min_computation_rank(const operation_state_ptrs& arguments);

    ElementwiseOperationState(const std::string& functor_name, const operation_state_ptrs& arguments);


    virtual DType dtype() const;

    virtual std::vector<int> bshape() const;

    virtual std::vector<operation_state_ptr> arguments() const;

    virtual void compute_node_compilation_info(int desired_computation_rank,
                                               const std::vector<int>& desired_computation_shape,
                                               std::vector<const ArrayOperationState*>* arrays,
                                               std::vector<const ScalarOperationState*>* scalars,
                                               node_to_info_t* node_to_info) const;

    virtual bool is_dim_collapsible_with_dim_minus_one(const int& dim) const;

    virtual operation_state_ptr collapse_dim_with_dim_minus_one(const int& dim) const;

    virtual operation_state_ptr transpose(const std::vector<int>& permutation) const;

    virtual std::string get_call_code_nd(const symbol_table_t& symbol_table, const node_to_info_t& node_to_info) const;

    virtual std::string prefix_code(const node_to_info_t& node_to_info) const;
};

struct CastOperationState : public ElementwiseOperationState {
    static const hash_t optype_hash;

    const DType dtype_;

    CastOperationState(DType dtype, const operation_state_ptr argument);

    virtual DType dtype() const;

    virtual void compute_node_compilation_info(
        int desired_computation_rank,
        const std::vector<int>& desired_computation_shape,
        std::vector<const ArrayOperationState*>* arrays,
        std::vector<const ScalarOperationState*>* scalars,
        node_to_info_t* node_to_info) const;
};



///////////////////////////////////////////////////////////////////////////////
//                   ELEMENTWISE OPERATION STATE                             //
///////////////////////////////////////////////////////////////////////////////

const hash_t ElementwiseOperationState::optype_hash = std::hash<std::string>()("ElementwiseOperationState");

int ElementwiseOperationState::compute_min_computation_rank(
        const operation_state_ptrs& arguments) {
    return std::accumulate(arguments.begin(),
                           arguments.end(),
                           0,
                           [](int so_far, operation_state_ptr op) {
                               return std::max(so_far, op->min_computation_rank_);
                           });
}

ElementwiseOperationState::ElementwiseOperationState(
    const std::string& functor_name,
    const operation_state_ptrs& arguments) :
        OperationState(compute_min_computation_rank(arguments)),
        functor_name_(functor_name),
        arguments_(arguments) {
}


DType ElementwiseOperationState::dtype() const {
    return arguments_[0]->dtype();
}

std::vector<int> ElementwiseOperationState::bshape() const {
    std::vector<std::vector<int>> arg_bshapes;
    for (auto& arg: arguments_) {
        arg_bshapes.emplace_back(arg->bshape());
    }
    return get_common_bshape(arg_bshapes);
}

std::vector<operation_state_ptr> ElementwiseOperationState::arguments() const { return arguments_; }

void ElementwiseOperationState::compute_node_compilation_info(
        int desired_computation_rank,
        const std::vector<int>& desired_computation_shape,
        std::vector<const ArrayOperationState*>* arrays,
        std::vector<const ScalarOperationState*>* scalars,
        node_to_info_t* node_to_info) const {
    (*node_to_info)[this].computation_rank = desired_computation_rank;
    for (auto& arg: arguments_) {
        arg->compute_node_compilation_info(desired_computation_rank, desired_computation_shape, arrays, scalars, node_to_info);
    }
    utils::Hasher hasher;
    hasher.add(optype_hash).add(desired_computation_rank).add(functor_name_);
    for (auto& arg: arguments_) {
        hasher.add(node_to_info->at(arg.get()).hash);
    }
    (*node_to_info)[this].hash = hasher.value();
}

bool ElementwiseOperationState::is_dim_collapsible_with_dim_minus_one(const int& dim) const {
    bool is_contig = true;
    for (auto& arg : arguments_) {
        is_contig = is_contig && arg->is_dim_collapsible_with_dim_minus_one(dim);
    }
    return is_contig;
}

operation_state_ptr ElementwiseOperationState::collapse_dim_with_dim_minus_one(const int& dim) const {
    operation_state_ptrs new_arguments;

    for (auto& arg : arguments_) {
        new_arguments.emplace_back(arg->collapse_dim_with_dim_minus_one(dim));
    }

    return std::make_shared<ElementwiseOperationState>(functor_name_, new_arguments);
}

operation_state_ptr ElementwiseOperationState::transpose(const std::vector<int>& permutation) const {
    operation_state_ptrs new_arguments;

    for (auto& arg : arguments_) {
        new_arguments.emplace_back(arg->transpose(permutation));
    }

    return std::make_shared<ElementwiseOperationState>(functor_name_, new_arguments);
}

std::string ElementwiseOperationState::get_call_code_nd(
        const symbol_table_t& symbol_table,
        const node_to_info_t& node_to_info) const {
    std::stringstream stream;
    stream << "element_wise_kernel<" << functor_name_ << ", " << dtype_to_cpp_name(dtype()) << ">(";

    for (int i = 0; i < arguments_.size(); ++i) {
        stream << arguments_[i]->get_call_code_nd(symbol_table, node_to_info)
               << (i + 1 == arguments_.size() ? "" : ", ");
    }
    stream << ")";
    return stream.str();
}

std::string ElementwiseOperationState::prefix_code(
        const node_to_info_t& node_to_info) const {
    return create_elementwise_kernel_caller(arguments_.size());
}


///////////////////////////////////////////////////////////////////////////////
//                       CAST OPERATION STATE                                //
///////////////////////////////////////////////////////////////////////////////

const hash_t CastOperationState::optype_hash = std::hash<std::string>()("CastOperationState");


CastOperationState::CastOperationState(
        DType dtype,
        const operation_state_ptr argument) :
    ElementwiseOperationState("functor::cast", {argument}),
    dtype_(dtype) {
}

void CastOperationState::compute_node_compilation_info(
        int desired_computation_rank,
        const std::vector<int>& desired_computation_shape,
        std::vector<const ArrayOperationState*>* arrays,
        std::vector<const ScalarOperationState*>* scalars,
        node_to_info_t* node_to_info) const {
    (*node_to_info)[this].computation_rank = desired_computation_rank;
    arguments_[0]->compute_node_compilation_info(desired_computation_rank, desired_computation_shape, arrays, scalars, node_to_info);

    (*node_to_info)[this].hash = utils::Hasher().add(optype_hash)
                                                .add(desired_computation_rank)
                                                .add(functor_name_)
                                                .add(node_to_info->at(arguments_[0].get()).hash)
                                                .add(dtype())
                                                .value();
}

DType CastOperationState::dtype() const {
    return dtype_;
}


///////////////////////////////////////////////////////////////////////////////
//                                OP2                                        //
///////////////////////////////////////////////////////////////////////////////


namespace op2 {
    Operation elementwise(const Operation& a,
                          const std::string& functor_name) {

        return Operation(std::make_shared<ElementwiseOperationState>(
            functor_name,
            operation_state_ptrs({a.state_})
        ));
    }

    Operation elementwise(
            const Operation& a,
            const Operation& b,
            const std::string& functor_name) {
        auto a_b = ensure_arguments_compatible(a, b);
        return Operation(std::make_shared<ElementwiseOperationState>(
            functor_name,
            operation_state_ptrs({std::get<0>(a_b).state_, std::get<1>(a_b).state_})
        ));
    }

    Operation astype(const Operation& a, DType type) {
        return Operation(std::make_shared<CastOperationState>(
            type,
            a.state_
        ));
    }

    std::tuple<Operation, Operation> ensure_arguments_compatible(
            const Operation& a, const Operation& b) {
        // perform type promotion:
        if (a.dtype() != b.dtype()) {
            auto new_type = type_promotion(a, b);
            if (a.dtype() == new_type) {
                // b's dtype is being promoted
                return std::tuple<Operation,Operation>(a, op2::astype(b, new_type));
            } else {

                // a's dtype is being promoted
                return std::tuple<Operation,Operation>(op2::astype(a, new_type), b);
            }
        } else {
            ASSERT2(ndim_compatible(a, b), "ranks don't match");
            return std::tuple<Operation,Operation>(a, b);
        }
    }
}
