#include "reducer_operation.h"
#include "dali/array/op2/elementwise_operation.h"
#include "dali/array/op2/rtc/rtc_expression.h"
#include "dali/utils/hash_utils.h"
#include "dali/utils/make_message.h"
#include "dali/array/op2/all_reduce_kernel_utils.h"

///////////////////////////////////////////////////////////////////////////////
//                    HEADERS                                                //
///////////////////////////////////////////////////////////////////////////////
namespace expression {
namespace rtc {
struct ReducerExpressionState : public RtcExpression {
    const std::shared_ptr<const RtcExpression> argument_;
    const std::string functor_name_;

    // MUST IMPLEMENT
    virtual hash_t optype_hash() const = 0;
    virtual std::string kernel_name() const = 0;

    // DO NOT REIMPLEMENT
    ReducerExpressionState(const std::string& functor_name,
                           std::shared_ptr<const RtcExpression> argument,
                           int min_computation_rank) :
        RtcExpression(min_computation_rank),
        functor_name_(functor_name),
        argument_(argument) {
    }

    virtual std::vector<std::shared_ptr<const ExpressionState>> arguments() const {
        return {argument_};
    }

    virtual std::string get_call_code_nd(const symbol_table_t& symbol_table,
                                         const node_to_info_t& node_to_info,
                                         memory::DeviceT device_type) const {
        int all_reduce_comp_rank = node_to_info.at(argument_.get()).computation_rank;
        return utils::make_message(
            kernel_name(), all_reduce_comp_rank,
            "d<", functor_name_, ", " , dtype_to_cpp_name(dtype()) , ">(",
            argument_->get_call_code_nd(symbol_table, node_to_info, device_type), ")");
    }
};  // struct ReducerExpressionState

struct AllReducerExpressionState : public ReducerExpressionState {
    static const hash_t optype_hash_cache_;

    virtual hash_t optype_hash() const {
        return optype_hash_cache_;
    }

    AllReducerExpressionState(const std::string& functor_name, std::shared_ptr<const RtcExpression> argument);

    virtual std::vector<int> bshape() const {
        return {};
    }
    virtual std::string name() const {
        return utils::make_message(
            "all_reduce<", functor_name_, ">"
        );
    }

    virtual DType dtype() const {
        return argument_->dtype();
    }

    virtual int ndim() const {
        return 0;
    }

    virtual void compute_node_compilation_info(int desired_computation_rank,
                                               const std::vector<int>& desired_computation_shape,
                                               std::vector<const RtcArrayWrapper*>* arrays,
                                               std::vector<const ScalarWrapper*>* scalars,
                                               node_to_info_t* node_to_info) const;

    virtual bool is_dim_collapsible_with_dim_minus_one(const int& dim) const {
        return true;
    }

    virtual std::shared_ptr<const RtcExpression> transpose(const std::vector<int>& permutation) const {
        return jit_shared_from_this();
    }

    virtual std::string prefix_code(const node_to_info_t& node_to_info, memory::DeviceT device_type) const;

    virtual std::string kernel_name() const {
        return "all_reduce_kernel_";
    }
};

struct AxisReducerExpressionState : public ReducerExpressionState {
    static const hash_t optype_hash_cache_;

    virtual hash_t optype_hash() const {
        return optype_hash_cache_;
    }

    AxisReducerExpressionState(const std::string& functor_name, std::shared_ptr<const RtcExpression> argument);

    virtual std::vector<int> bshape() const;

    virtual DType dtype() const {
        return argument_->dtype();
    }

    virtual std::string name() const {
        return utils::make_message(
            "axis_reduce<", functor_name_, ">"
        );
    }

    virtual int ndim() const {
        return std::max(argument_->ndim() - 1, 0);
    }

    virtual void compute_node_compilation_info(
        int desired_computation_rank,
        const std::vector<int>& desired_computation_shape,
        std::vector<const RtcArrayWrapper*>* arrays,
        std::vector<const ScalarWrapper*>* scalars,
        node_to_info_t* node_to_info) const;

    virtual bool is_dim_collapsible_with_dim_minus_one(const int& dim) const;

    virtual std::shared_ptr<const RtcExpression> collapse_dim_with_dim_minus_one(const int& dim) const;

    virtual std::shared_ptr<const RtcExpression> transpose(const std::vector<int>& permutation) const;

    virtual std::string prefix_code(const node_to_info_t& node_to_info, memory::DeviceT device_type) const;

    virtual std::string kernel_name() const {
        return "axis_reduce_kernel_";
    }
};


struct ArgumentAllReducerExpressionState : public AllReducerExpressionState {
    static const hash_t optype_hash_cache_;

    using AllReducerExpressionState::AllReducerExpressionState;

    virtual hash_t optype_hash() const {
        return optype_hash_cache_;
    }

    virtual DType dtype() const {
        return DTYPE_INT32;
    }

    virtual std::string name() const {
        return utils::make_message(
            "argument_all_reduce<", functor_name_, ">"
        );
    }

    virtual std::string prefix_code(const node_to_info_t& node_to_info, memory::DeviceT device_type) const;

    virtual std::string kernel_name() const {
        return "argument_all_reduce_kernel_";
    }
};


struct ArgumentAxisReducerExpressionState : public AxisReducerExpressionState {
    static const hash_t optype_hash_cache_;

    using AxisReducerExpressionState::AxisReducerExpressionState;

    virtual hash_t optype_hash() const {
        return optype_hash_cache_;
    }

    virtual DType dtype() const {
        return DTYPE_INT32;
    }

    virtual std::string name() const {
        return utils::make_message(
            "argument_axis_reduce<", functor_name_, ">"
        );
    }

    virtual std::string prefix_code(const node_to_info_t& node_to_info, memory::DeviceT device_type) const;

    virtual std::shared_ptr<const RtcExpression> collapse_dim_with_dim_minus_one(const int& dim) const;

    virtual std::shared_ptr<const RtcExpression> transpose(const std::vector<int>& permutation) const;

    virtual std::string kernel_name() const {
        return "argument_axis_reduce_kernel_";
    }
};


///////////////////////////////////////////////////////////////////////////////
//                    ALL REDUCER OPERATION STATE                            //
///////////////////////////////////////////////////////////////////////////////


const hash_t AxisReducerExpressionState::optype_hash_cache_ = std::hash<std::string>()("AxisReducerExpressionState");

AllReducerExpressionState::AllReducerExpressionState(
        const std::string& functor_name,
        std::shared_ptr<const RtcExpression> argument) :
    ReducerExpressionState(functor_name, argument, 1) {
}

void AllReducerExpressionState::compute_node_compilation_info(
        int desired_computation_rank,
        const std::vector<int>& desired_computation_shape,
        std::vector<const RtcArrayWrapper*>* arrays,
        std::vector<const ScalarWrapper*>* scalars,
        node_to_info_t* node_to_info) const {
    (*node_to_info)[this].computation_rank = desired_computation_rank;
    argument_->compute_node_compilation_info(argument_->min_computation_rank_, argument_->shape(), arrays, scalars, node_to_info);
    (*node_to_info)[this].hash = utils::Hasher().add(optype_hash())
                                                .add(desired_computation_rank)
                                                .add(functor_name_)
                                                .add(node_to_info->at(argument_.get()).hash)
                                                .value();
}

std::string AllReducerExpressionState::prefix_code(
        const node_to_info_t& node_to_info,
        memory::DeviceT device_type) const {
    return create_all_reduce_kernel_caller(
        node_to_info.at(argument_.get()).computation_rank,
        node_to_info.at(this).computation_rank
    );
}



///////////////////////////////////////////////////////////////////////////////
//                    AXIS REDUCER OPERATION STATE                           //
///////////////////////////////////////////////////////////////////////////////

const hash_t ArgumentAllReducerExpressionState::optype_hash_cache_ = std::hash<std::string>()("ArgumentAllReducerExpressionState");

AxisReducerExpressionState::AxisReducerExpressionState(
        const std::string& functor_name,
        std::shared_ptr<const RtcExpression> argument) :
    ReducerExpressionState(functor_name, argument, std::max(argument->min_computation_rank_ - 1, 1)) {
}

std::vector<int> AxisReducerExpressionState::bshape() const {
    auto result = argument_->bshape();
    result.pop_back();
    return result;
}


void AxisReducerExpressionState::compute_node_compilation_info(
        int desired_computation_rank,
        const std::vector<int>& desired_computation_shape,
        std::vector<const RtcArrayWrapper*>* arrays,
        std::vector<const ScalarWrapper*>* scalars,
        node_to_info_t* node_to_info) const {
    (*node_to_info)[this].computation_rank = desired_computation_rank;

    auto desired_argument_shape = desired_computation_shape;
    desired_argument_shape.emplace_back(argument_->shape().back());


    argument_->compute_node_compilation_info(desired_computation_rank + 1, desired_argument_shape, arrays, scalars, node_to_info);

    (*node_to_info)[this].hash = utils::Hasher().add(optype_hash())
                                                .add(desired_computation_rank)
                                                .add(functor_name_)
                                                .add(node_to_info->at(argument_.get()).hash)
                                                .value();
}

bool AxisReducerExpressionState::is_dim_collapsible_with_dim_minus_one(
        const int& dim) const {
    return argument_->is_dim_collapsible_with_dim_minus_one(dim - 1);;
}

std::shared_ptr<const RtcExpression> AxisReducerExpressionState::collapse_dim_with_dim_minus_one(
        const int& dim) const {
    return std::make_shared<AxisReducerExpressionState>(
        functor_name_,
        argument_->collapse_dim_with_dim_minus_one(dim - 1)
    );
}

std::shared_ptr<const RtcExpression> AxisReducerExpressionState::transpose(
        const std::vector<int>& permutation) const {
    auto new_permutation = permutation;
    // add last dim of tensor with rank (permutation.size() + 1)
    new_permutation.emplace_back(permutation.size());

    return std::make_shared<AxisReducerExpressionState>(
        functor_name_,
        argument_->transpose(new_permutation)
    );
}

std::string AxisReducerExpressionState::prefix_code(
        const node_to_info_t& node_to_info,
        memory::DeviceT device_type) const {
    return create_axis_reduce_kernel_caller(node_to_info.at(argument_.get()).computation_rank);
}


///////////////////////////////////////////////////////////////////////////////
//                ARGUMENT ALL REDUCER OPERATION STATE                       //
///////////////////////////////////////////////////////////////////////////////

std::string ArgumentAllReducerExpressionState::prefix_code(
        const node_to_info_t& node_to_info,
        memory::DeviceT device_type) const {
    return create_argument_all_reduce_kernel_caller(
        node_to_info.at(argument_.get()).computation_rank,
        node_to_info.at(this).computation_rank
    );
}


///////////////////////////////////////////////////////////////////////////////
//         ARGUMENT AXIS REDUCER OPERATION STATE                             //
///////////////////////////////////////////////////////////////////////////////

const hash_t ArgumentAxisReducerExpressionState::optype_hash_cache_ = std::hash<std::string>()("ArgumentAxisReducerExpressionState");

std::string ArgumentAxisReducerExpressionState::prefix_code(
        const node_to_info_t& node_to_info,
        memory::DeviceT device_type) const {
    return create_argument_axis_reduce_kernel_caller(node_to_info.at(argument_.get()).computation_rank);
}

std::shared_ptr<const RtcExpression> ArgumentAxisReducerExpressionState::collapse_dim_with_dim_minus_one(
        const int& dim) const {
    return std::make_shared<ArgumentAxisReducerExpressionState>(
        functor_name_,
        argument_->collapse_dim_with_dim_minus_one(dim - 1)
    );
}

std::shared_ptr<const RtcExpression> ArgumentAxisReducerExpressionState::transpose(
        const std::vector<int>& permutation) const {
    auto new_permutation = permutation;
    // add last dim of tensor with rank (permutation.size() + 1)
    new_permutation.emplace_back(permutation.size());

    return std::make_shared<ArgumentAxisReducerExpressionState>(
        functor_name_,
        argument_->transpose(new_permutation)
    );
}

const hash_t AllReducerExpressionState::optype_hash_cache_ =
        std::hash<std::string>()("AllReducerExpressionState");

}  // namespace rtc
}  // namespace expression


namespace op {
    expression::Expression all_reduce(
            const expression::Expression& a,
            const std::string& reducer_name) {
        return expression::Expression(std::make_shared<expression::rtc::AllReducerExpressionState>(
            reducer_name,
            a.state_->as_jit()
        ));
    }

    expression::Expression axis_reduce(
            const expression::Expression& a,
            const std::string& reducer_name,
            const std::vector<int>& axes) {
        if (axes.size() == 0) return a;
        int ndim = a.ndim();
        if (ndim == 0) return a;
        std::vector<int> normalized_axes(axes);
        for (auto& axis : normalized_axes) {
            if (axis < 0) {
                if (ndim == 0) {
                    axis = axis + 1;
                } else {
                    axis = axis + ndim;
                }
            }
            ASSERT2(axis >= 0 && (axis < ndim || ndim == 0 && axis == ndim),
                utils::make_message(
                    "Reduction axis must strictly positive and less than the "
                    "number of dimensions of the input (got axis=", axes[0], ","
                    " ndim=", ndim, ")."
                )
            );
        }
        // now look to see what kind of a reduction this is:
        std::vector<bool> reduced_dims(ndim, false);
        std::sort(normalized_axes.begin(), normalized_axes.end());
        for (auto& axis : normalized_axes) {
            ASSERT2(!reduced_dims[axis], utils::make_message("axis_reduce "
                "received duplicate axes to operate on (axis=", axis,
                " axes=", axes, ")."
            ));
            reduced_dims[axis] = true;
        }
        // all axes are present:
        if (normalized_axes.size() == ndim) {
            return all_reduce(a, reducer_name);
        }
        int num_low_dims = 0;
        for (int i = reduced_dims.size() - 1; i >= 0; --i) {
            if (reduced_dims[i]) {
                ++num_low_dims;
            } else {
                break;
            }
        }
        bool all_reductions_are_low_dim = num_low_dims == normalized_axes.size();
        auto res = a.state_->as_jit();

        if (!all_reductions_are_low_dim) {
            std::vector<int> new_axes_order;
            for (int i = 0; i < reduced_dims.size(); ++i) {
                if (!reduced_dims[i]) {
                    new_axes_order.emplace_back(i);
                }
            }
            for (int i = 0; i < reduced_dims.size(); ++i) {
                if (reduced_dims[i]) {
                    new_axes_order.emplace_back(i);
                }
            }
            res = res->transpose(new_axes_order);
        }
        int num_low_axes_to_reduce = normalized_axes.size();
        if (num_low_axes_to_reduce > 0) {
            int axes_used_up = 0;
            int collapsed_ndim = ndim - 1;
            for (int axes_used_up = 0; axes_used_up < num_low_axes_to_reduce; ++axes_used_up) {
                if (num_low_axes_to_reduce - axes_used_up == 1) {
                    res = std::make_shared<expression::rtc::AxisReducerExpressionState>(
                        reducer_name,
                        res
                    );
                } else {
                    if (res->is_dim_collapsible_with_dim_minus_one(collapsed_ndim)) {
                        res = res->collapse_dim_with_dim_minus_one(collapsed_ndim);
                    } else {
                        res = std::make_shared<expression::rtc::AxisReducerExpressionState>(
                            reducer_name,
                            res
                        );
                    }
                }
                --collapsed_ndim;
            }
        }
        return expression::Expression(res);
    }

    expression::Expression argument_all_reduce(
            const expression::Expression& a,
            const std::string& reducer_name) {
        return expression::Expression(std::make_shared<expression::rtc::ArgumentAllReducerExpressionState>(
            reducer_name,
            a.state_->as_jit()
        ));
    }

    expression::Expression argument_axis_reduce(
            const expression::Expression& a,
            const std::string& reducer_name,
            const int& axis) {
        int ndim = a.ndim();
        if (ndim == 0) return expression::Expression(0);
        int normalized_axis = axis;
        if (normalized_axis < 0) normalized_axis = normalized_axis + a.ndim();
        ASSERT2(normalized_axis >= 0 && (normalized_axis < ndim || ndim == 0 && normalized_axis == ndim),
            utils::make_message(
                "Reduction axis must strictly positive and less than the "
                "number of dimensions of the input (got axis=", normalized_axis, ","
                " ndim=", ndim, ")."
            )
        );
        if (ndim == 1) return argument_all_reduce(a, reducer_name);

        auto res = a.state_->as_jit();
        if (normalized_axis != ndim - 1) {
            std::vector<int> axes;
            for (int i = 0; i < ndim; i++) {
                axes.emplace_back(i);
            }
            axes[axes.size() - 1] = normalized_axis;
            axes[normalized_axis] = axes.size() - 1;
            res = res->transpose(axes);
        }
        return expression::Expression(std::make_shared<expression::rtc::ArgumentAxisReducerExpressionState>(
            reducer_name,
            res
        ));
    }
}  // namespace op2
