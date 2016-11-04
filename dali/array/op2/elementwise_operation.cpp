#include "elementwise_operation.h"

#include <vector>
#include "dali/utils/hash_utils.h"
#include "dali/utils/assert2.h"
#include "dali/array/op2/rtc_utils.h"
#include "dali/array/op2/elementwise_kernel_utils.h"

DType type_promotion(const Expression& a, const Expression& b) {
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

bool ndim_compatible(const Expression& a, const Expression& b) {
    int a_ndim = a.ndim();
    int b_ndim = b.ndim();
    return a_ndim == 0 || b_ndim == 0 || a_ndim == b_ndim;
}


///////////////////////////////////////////////////////////////////////////////
//                       HEADERS                                             //
///////////////////////////////////////////////////////////////////////////////

struct ElementwiseExpressionState : public RtcExpression {
    static const hash_t optype_hash;

    const std::vector<std::shared_ptr<const RtcExpression>> arguments_;
    const std::string functor_name_;

    static int compute_min_computation_rank(const std::vector<std::shared_ptr<const RtcExpression>>& arguments);

    ElementwiseExpressionState(const std::string& functor_name, const std::vector<std::shared_ptr<const RtcExpression>>& arguments);


    virtual DType dtype() const;
    virtual std::string name() const;

    virtual std::vector<int> bshape() const;

    virtual std::vector<std::shared_ptr<const ExpressionState>> arguments() const;

    virtual void compute_node_compilation_info(int desired_computation_rank,
                                               const std::vector<int>& desired_computation_shape,
                                               std::vector<const ArrayWrapper*>* arrays,
                                               std::vector<const ScalarWrapper*>* scalars,
                                               node_to_info_t* node_to_info) const;

    virtual bool is_dim_collapsible_with_dim_minus_one(const int& dim) const;

    virtual std::shared_ptr<const RtcExpression> collapse_dim_with_dim_minus_one(const int& dim) const;

    virtual std::shared_ptr<const RtcExpression> transpose(const std::vector<int>& permutation) const;

    virtual std::string get_call_code_nd(const symbol_table_t& symbol_table, const node_to_info_t& node_to_info, memory::DeviceT device_type) const;

    virtual std::string prefix_code(const node_to_info_t& node_to_info, memory::DeviceT device_type) const;
};


///////////////////////////////////////////////////////////////////////////////
//                   ELEMENTWISE OPERATION STATE                             //
///////////////////////////////////////////////////////////////////////////////

const hash_t ElementwiseExpressionState::optype_hash = std::hash<std::string>()("ElementwiseExpressionState");

int ElementwiseExpressionState::compute_min_computation_rank(
        const std::vector<std::shared_ptr<const RtcExpression>>& arguments) {
    return std::accumulate(arguments.begin(),
                           arguments.end(),
                           0,
                           [](int so_far, std::shared_ptr<const RtcExpression> op) {
                               return std::max(so_far, op->min_computation_rank_);
                           });
}

ElementwiseExpressionState::ElementwiseExpressionState(
    const std::string& functor_name,
    const std::vector<std::shared_ptr<const RtcExpression>>& arguments) :
        RtcExpression(compute_min_computation_rank(arguments)),
        functor_name_(functor_name),
        arguments_(arguments) {
}


DType ElementwiseExpressionState::dtype() const {
    return arguments_[0]->dtype();
}

std::string ElementwiseExpressionState::name() const {
    return functor_name_;
}

std::vector<int> ElementwiseExpressionState::bshape() const {
    std::vector<std::vector<int>> arg_bshapes;
    for (auto& arg: arguments_) {
        arg_bshapes.emplace_back(arg->bshape());
    }
    return get_common_bshape(arg_bshapes);
}

std::vector<std::shared_ptr<const ExpressionState>> ElementwiseExpressionState::arguments() const {
    return std::vector<std::shared_ptr<const ExpressionState>>(arguments_.begin(), arguments_.end());
}

void ElementwiseExpressionState::compute_node_compilation_info(
        int desired_computation_rank,
        const std::vector<int>& desired_computation_shape,
        std::vector<const ArrayWrapper*>* arrays,
        std::vector<const ScalarWrapper*>* scalars,
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

bool ElementwiseExpressionState::is_dim_collapsible_with_dim_minus_one(const int& dim) const {
    bool is_contig = true;
    for (auto& arg : arguments_) {
        is_contig = is_contig && arg->is_dim_collapsible_with_dim_minus_one(dim);
    }
    return is_contig;
}

std::shared_ptr<const RtcExpression> ElementwiseExpressionState::collapse_dim_with_dim_minus_one(const int& dim) const {
    std::vector<std::shared_ptr<const RtcExpression>> new_arguments;

    for (auto& arg : arguments_) {
        new_arguments.emplace_back(arg->collapse_dim_with_dim_minus_one(dim));
    }

    return std::make_shared<ElementwiseExpressionState>(functor_name_, new_arguments);
}

std::shared_ptr<const RtcExpression> ElementwiseExpressionState::transpose(const std::vector<int>& permutation) const {
    std::vector<std::shared_ptr<const RtcExpression>> new_arguments;

    for (auto& arg : arguments_) {
        new_arguments.emplace_back(arg->transpose(permutation));
    }

    return std::make_shared<ElementwiseExpressionState>(functor_name_, new_arguments);
}

std::string ElementwiseExpressionState::get_call_code_nd(
        const symbol_table_t& symbol_table,
        const node_to_info_t& node_to_info,
        memory::DeviceT device_type) const {
    std::stringstream stream;
    stream << "element_wise_kernel<" << functor_name_ << ", " << dtype_to_cpp_name(dtype()) << ">(";

    for (int i = 0; i < arguments_.size(); ++i) {
        stream << arguments_[i]->get_call_code_nd(symbol_table, node_to_info, device_type)
               << (i + 1 == arguments_.size() ? "" : ", ");
    }
    stream << ")";
    return stream.str();
}

std::string ElementwiseExpressionState::prefix_code(
        const node_to_info_t& node_to_info,
        memory::DeviceT device_type) const {
    return create_elementwise_kernel_caller(arguments_.size());
}


///////////////////////////////////////////////////////////////////////////////
//                       CAST OPERATION STATE                                //
///////////////////////////////////////////////////////////////////////////////

struct CastExpressionState : public ElementwiseExpressionState {
    static const hash_t optype_hash;

    const DType dtype_;

    CastExpressionState(DType dtype, const std::shared_ptr<const RtcExpression> argument) :
        ElementwiseExpressionState("functor::cast", {argument}),
        dtype_(dtype) {
    }

    virtual DType dtype() const {
        return dtype_;
    }

    virtual void compute_node_compilation_info(
        int desired_computation_rank,
        const std::vector<int>& desired_computation_shape,
        std::vector<const ArrayWrapper*>* arrays,
        std::vector<const ScalarWrapper*>* scalars,
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
};

const hash_t CastExpressionState::optype_hash = std::hash<std::string>()("CastExpressionState");

struct RoundExpressionState : public ElementwiseExpressionState {
    static const hash_t optype_hash;

    RoundExpressionState(const std::shared_ptr<const RtcExpression> argument) :
        ElementwiseExpressionState("functor::round", {argument}) {
    }

    virtual DType dtype() const {
        return DTYPE_INT32;
    }

    virtual void compute_node_compilation_info(
        int desired_computation_rank,
        const std::vector<int>& desired_computation_shape,
        std::vector<const ArrayWrapper*>* arrays,
        std::vector<const ScalarWrapper*>* scalars,
        node_to_info_t* node_to_info) const {
        (*node_to_info)[this].computation_rank = desired_computation_rank;
        arguments_[0]->compute_node_compilation_info(desired_computation_rank, desired_computation_shape, arrays, scalars, node_to_info);

        (*node_to_info)[this].hash = utils::Hasher().add(optype_hash)
                                                    .add(desired_computation_rank)
                                                    .add(functor_name_)
                                                    .add(node_to_info->at(arguments_[0].get()).hash)
                                                    .value();
    }
};

const hash_t RoundExpressionState::optype_hash = std::hash<std::string>()("RoundExpressionState");

///////////////////////////////////////////////////////////////////////////////
//                                OP2                                        //
///////////////////////////////////////////////////////////////////////////////


namespace op {
    Expression elementwise(const Expression& a,
                          const std::string& functor_name) {

        return Expression(std::make_shared<ElementwiseExpressionState>(
            functor_name,
            std::vector<std::shared_ptr<const RtcExpression>>({a.state_->as_jit()})
        ));
    }

    Expression elementwise(
            const Expression& a,
            const Expression& b,
            const std::string& functor_name) {
        auto a_b = ensure_arguments_compatible(a, b);
        return Expression(std::make_shared<ElementwiseExpressionState>(
            functor_name,
            std::vector<std::shared_ptr<const RtcExpression>>({std::get<0>(a_b).state_->as_jit(), std::get<1>(a_b).state_->as_jit()})
        ));
    }

    Expression astype(const Expression& a, DType type) {
        if (type == DTYPE_INT32) {
            return round(a);
        } else {
            return unsafe_cast(a, type);
        }
    }

    Expression unsafe_cast(const Expression& a, DType type) {
        return Expression(std::make_shared<CastExpressionState>(
            type,
            a.state_->as_jit()
        ));
    }

    Expression round(const Expression& a) {
        return Expression(std::make_shared<RoundExpressionState>(
            a.state_->as_jit()
        ));
    }

    std::tuple<Expression, Expression> ensure_arguments_compatible(
            const Expression& a, const Expression& b) {
        // perform type promotion:
        if (a.dtype() != b.dtype()) {
            auto new_type = type_promotion(a, b);
            if (a.dtype() == new_type) {
                // b's dtype is being promoted
                return std::tuple<Expression,Expression>(a, op::astype(b, new_type));
            } else {

                // a's dtype is being promoted
                return std::tuple<Expression,Expression>(op::astype(a, new_type), b);
            }
        } else {
            ASSERT2(ndim_compatible(a, b), "ranks don't match");
            return std::tuple<Expression,Expression>(a, b);
        }
    }
}
