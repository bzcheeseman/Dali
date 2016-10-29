#include "reducer_operation.h"
#include "dali/array/op2/elementwise_operation.h"
#include "dali/utils/hash_utils.h"
#include "dali/utils/make_message.h"
#include "dali/array/op2/all_reduce_kernel_utils.h"

///////////////////////////////////////////////////////////////////////////////
//                    HEADERS                                                //
///////////////////////////////////////////////////////////////////////////////

struct ReducerOperationState : public JITOperationState {
    const std::shared_ptr<const JITOperationState> argument_;
    const std::string functor_name_;

    virtual hash_t optype_hash() const = 0;

    ReducerOperationState(const std::string& functor_name, std::shared_ptr<const JITOperationState> argument, int min_computation_rank);

    virtual std::vector<operation_state_ptr> arguments() const;

    virtual std::string get_call_code_nd(const symbol_table_t& symbol_table, const node_to_info_t& node_to_info, memory::DeviceT device_type) const;

    virtual std::string kernel_name() const = 0;
};

struct AllReducerOperationState : public ReducerOperationState {
    static const hash_t optype_hash_cache_;

    virtual hash_t optype_hash() const;

    AllReducerOperationState(const std::string& functor_name, std::shared_ptr<const JITOperationState> argument);

    virtual std::vector<int> bshape() const;
    virtual std::string name() const {
        return utils::make_message(
            "all_reduce<", functor_name_, ">"
        );
    }

    virtual DType dtype() const;

    virtual int ndim() const;

    virtual void compute_node_compilation_info(int desired_computation_rank,
                                               const std::vector<int>& desired_computation_shape,
                                               std::vector<const ArrayOperationState*>* arrays,
                                               std::vector<const ScalarOperationState*>* scalars,
                                               node_to_info_t* node_to_info) const;

    virtual bool is_dim_collapsible_with_dim_minus_one(const int& dim) const;

    virtual std::shared_ptr<const JITOperationState> transpose(const std::vector<int>& permutation) const;

    virtual std::string prefix_code(const node_to_info_t& node_to_info, memory::DeviceT device_type) const;

    virtual std::string kernel_name() const;
};

struct AxisReducerOperationState : public ReducerOperationState {
    static const hash_t optype_hash_cache_;

    virtual hash_t optype_hash() const;

    AxisReducerOperationState(const std::string& functor_name, std::shared_ptr<const JITOperationState> argument);

    virtual std::vector<int> bshape() const;

    virtual DType dtype() const;
    virtual std::string name() const {
        return utils::make_message(
            "axis_reduce<", functor_name_, ">"
        );
    }

    virtual int ndim() const;

    virtual void compute_node_compilation_info(
        int desired_computation_rank,
        const std::vector<int>& desired_computation_shape,
        std::vector<const ArrayOperationState*>* arrays,
        std::vector<const ScalarOperationState*>* scalars,
        node_to_info_t* node_to_info) const;

    virtual bool is_dim_collapsible_with_dim_minus_one(const int& dim) const;

    virtual std::shared_ptr<const JITOperationState> collapse_dim_with_dim_minus_one(const int& dim) const;

    virtual std::shared_ptr<const JITOperationState> transpose(const std::vector<int>& permutation) const;

    virtual std::string prefix_code(const node_to_info_t& node_to_info, memory::DeviceT device_type) const;

    virtual std::string kernel_name() const;
};


struct ArgumentAllReducerOperationState : public AllReducerOperationState {
    static const hash_t optype_hash_cache_;

    using AllReducerOperationState::AllReducerOperationState;

    virtual hash_t optype_hash() const;

    virtual DType dtype() const;
    virtual std::string name() const {
        return utils::make_message(
            "argument_all_reduce<", functor_name_, ">"
        );
    }

    virtual std::string prefix_code(const node_to_info_t& node_to_info, memory::DeviceT device_type) const;

    virtual std::string kernel_name() const;
};


struct ArgumentAxisReducerOperationState : public AxisReducerOperationState {
    static const hash_t optype_hash_cache_;

    using AxisReducerOperationState::AxisReducerOperationState;

    virtual hash_t optype_hash() const;

    virtual DType dtype() const;
    virtual std::string name() const {
        return utils::make_message(
            "argument_axis_reduce<", functor_name_, ">"
        );
    }

    virtual std::string prefix_code(const node_to_info_t& node_to_info, memory::DeviceT device_type) const;

    virtual std::shared_ptr<const JITOperationState> collapse_dim_with_dim_minus_one(const int& dim) const;

    virtual std::shared_ptr<const JITOperationState> transpose(const std::vector<int>& permutation) const;

    virtual std::string kernel_name() const;
};


///////////////////////////////////////////////////////////////////////////////
//                    ALL REDUCER OPERATION STATE                            //
///////////////////////////////////////////////////////////////////////////////


const hash_t AxisReducerOperationState::optype_hash_cache_ = std::hash<std::string>()("AxisReducerOperationState");

hash_t AllReducerOperationState::optype_hash() const {
    return optype_hash_cache_;
}

AllReducerOperationState::AllReducerOperationState(
        const std::string& functor_name,
        std::shared_ptr<const JITOperationState> argument) :
    ReducerOperationState(functor_name, argument, 1) {
}

std::vector<int> AllReducerOperationState::bshape() const {
    return {};
}

DType AllReducerOperationState::dtype() const {
    return argument_->dtype();
}

int AllReducerOperationState::ndim() const {
    return 0;
}

void AllReducerOperationState::compute_node_compilation_info(
        int desired_computation_rank,
        const std::vector<int>& desired_computation_shape,
        std::vector<const ArrayOperationState*>* arrays,
        std::vector<const ScalarOperationState*>* scalars,
        node_to_info_t* node_to_info) const {
    (*node_to_info)[this].computation_rank = desired_computation_rank;
    argument_->compute_node_compilation_info(argument_->min_computation_rank_, argument_->shape(), arrays, scalars, node_to_info);
    (*node_to_info)[this].hash = utils::Hasher().add(optype_hash())
                                                .add(desired_computation_rank)
                                                .add(functor_name_)
                                                .add(node_to_info->at(argument_.get()).hash)
                                                .value();
}


bool AllReducerOperationState::is_dim_collapsible_with_dim_minus_one(
        const int& dim) const {
    return true;
}

std::shared_ptr<const JITOperationState> AllReducerOperationState::transpose(
        const std::vector<int>& permutation) const {
    return jit_shared_from_this();
}

std::string AllReducerOperationState::prefix_code(
        const node_to_info_t& node_to_info,
        memory::DeviceT device_type) const {
    return create_all_reduce_kernel_caller(
        node_to_info.at(argument_.get()).computation_rank,
        node_to_info.at(this).computation_rank
    );
}

std::string AllReducerOperationState::kernel_name() const {
    return "all_reduce_kernel_";
}



///////////////////////////////////////////////////////////////////////////////
//                    AXIS REDUCER OPERATION STATE                           //
///////////////////////////////////////////////////////////////////////////////

const hash_t ArgumentAllReducerOperationState::optype_hash_cache_ = std::hash<std::string>()("ArgumentAllReducerOperationState");

hash_t AxisReducerOperationState::optype_hash() const {
    return optype_hash_cache_;
}


AxisReducerOperationState::AxisReducerOperationState(
        const std::string& functor_name,
        std::shared_ptr<const JITOperationState> argument) :
    ReducerOperationState(functor_name, argument, std::max(argument->min_computation_rank_ - 1, 1)) {
}

std::vector<int> AxisReducerOperationState::bshape() const {
    auto result = argument_->bshape();
    result.pop_back();
    return result;
}

DType AxisReducerOperationState::dtype() const {
    return argument_->dtype();
}

int AxisReducerOperationState::ndim() const {
    return std::max(argument_->ndim() - 1, 0);
}

void AxisReducerOperationState::compute_node_compilation_info(
        int desired_computation_rank,
        const std::vector<int>& desired_computation_shape,
        std::vector<const ArrayOperationState*>* arrays,
        std::vector<const ScalarOperationState*>* scalars,
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

bool AxisReducerOperationState::is_dim_collapsible_with_dim_minus_one(
        const int& dim) const {
    return argument_->is_dim_collapsible_with_dim_minus_one(dim - 1);;
}

std::shared_ptr<const JITOperationState> AxisReducerOperationState::collapse_dim_with_dim_minus_one(
        const int& dim) const {
    return std::make_shared<AxisReducerOperationState>(
        functor_name_,
        argument_->collapse_dim_with_dim_minus_one(dim - 1)
    );
}

std::shared_ptr<const JITOperationState> AxisReducerOperationState::transpose(
        const std::vector<int>& permutation) const {
    auto new_permutation = permutation;
    // add last dim of tensor with rank (permutation.size() + 1)
    new_permutation.emplace_back(permutation.size());

    return std::make_shared<AxisReducerOperationState>(
        functor_name_,
        argument_->transpose(new_permutation)
    );
}

std::string AxisReducerOperationState::prefix_code(
        const node_to_info_t& node_to_info,
        memory::DeviceT device_type) const {
    return create_axis_reduce_kernel_caller(node_to_info.at(argument_.get()).computation_rank);
}

std::string AxisReducerOperationState::kernel_name() const {
    return "axis_reduce_kernel_";
}


///////////////////////////////////////////////////////////////////////////////
//                ARGUMENT ALL REDUCER OPERATION STATE                       //
///////////////////////////////////////////////////////////////////////////////



hash_t ArgumentAllReducerOperationState::optype_hash() const {
    return optype_hash_cache_;
}

DType ArgumentAllReducerOperationState::dtype() const {
    return DTYPE_INT32;
}

std::string ArgumentAllReducerOperationState::prefix_code(
        const node_to_info_t& node_to_info,
        memory::DeviceT device_type) const {
    return create_argument_all_reduce_kernel_caller(
        node_to_info.at(argument_.get()).computation_rank,
        node_to_info.at(this).computation_rank
    );
}

std::string ArgumentAllReducerOperationState::kernel_name() const {
    return "argument_all_reduce_kernel_";
}


///////////////////////////////////////////////////////////////////////////////
//         ARGUMENT AXIS REDUCER OPERATION STATE                             //
///////////////////////////////////////////////////////////////////////////////

const hash_t ArgumentAxisReducerOperationState::optype_hash_cache_ = std::hash<std::string>()("ArgumentAxisReducerOperationState");

hash_t ArgumentAxisReducerOperationState::optype_hash() const {
    return optype_hash_cache_;
}

DType ArgumentAxisReducerOperationState::dtype() const {
    return DTYPE_INT32;
}

std::string ArgumentAxisReducerOperationState::prefix_code(
        const node_to_info_t& node_to_info,
        memory::DeviceT device_type) const {
    return create_argument_axis_reduce_kernel_caller(node_to_info.at(argument_.get()).computation_rank);
}

std::shared_ptr<const JITOperationState> ArgumentAxisReducerOperationState::collapse_dim_with_dim_minus_one(
        const int& dim) const {
    return std::make_shared<ArgumentAxisReducerOperationState>(
        functor_name_,
        argument_->collapse_dim_with_dim_minus_one(dim - 1)
    );
}

std::shared_ptr<const JITOperationState> ArgumentAxisReducerOperationState::transpose(
        const std::vector<int>& permutation) const {
    auto new_permutation = permutation;
    // add last dim of tensor with rank (permutation.size() + 1)
    new_permutation.emplace_back(permutation.size());

    return std::make_shared<ArgumentAxisReducerOperationState>(
        functor_name_,
        argument_->transpose(new_permutation)
    );
}

std::string ArgumentAxisReducerOperationState::kernel_name() const {
    return "argument_axis_reduce_kernel_";
}


///////////////////////////////////////////////////////////////////////////////
//                       REDUCER OPERATION STATE                             //
///////////////////////////////////////////////////////////////////////////////

const hash_t AllReducerOperationState::optype_hash_cache_ =
        std::hash<std::string>()("AllReducerOperationState");

ReducerOperationState::ReducerOperationState(
        const std::string& functor_name,
        std::shared_ptr<const JITOperationState> argument,
        int min_computation_rank) :
    JITOperationState(min_computation_rank),
    functor_name_(functor_name),
    argument_(argument) {

}

std::vector<operation_state_ptr> ReducerOperationState::arguments() const {
    return {argument_};
}

std::string ReducerOperationState::get_call_code_nd(
        const symbol_table_t& symbol_table,
        const node_to_info_t& node_to_info,
        memory::DeviceT device_type) const {
    int all_reduce_comp_rank = node_to_info.at(argument_.get()).computation_rank;
    return utils::make_message(
        kernel_name(), all_reduce_comp_rank,
        "d<", functor_name_, ", " , dtype_to_cpp_name(dtype()) , ">(",
        argument_->get_call_code_nd(symbol_table, node_to_info, device_type), ")");

}


namespace op {
    Operation all_reduce(const Operation& a,
                         const std::string& reducer_name) {
        return Operation(std::make_shared<AllReducerOperationState>(
            reducer_name,
            a.state_->as_jit()
        ));
    }

    Operation axis_reduce(const Operation& a,
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
                    res = std::make_shared<AxisReducerOperationState>(
                        reducer_name,
                        res
                    );
                } else {
                    if (res->is_dim_collapsible_with_dim_minus_one(collapsed_ndim)) {
                        res = res->collapse_dim_with_dim_minus_one(collapsed_ndim);
                    } else {
                        res = std::make_shared<AxisReducerOperationState>(
                            reducer_name,
                            res
                        );
                    }
                }
                --collapsed_ndim;
            }
        }
        return Operation(res);
    }

    Operation argument_all_reduce(const Operation& a,
                                 const std::string& reducer_name) {
        return Operation(std::make_shared<ArgumentAllReducerOperationState>(
            reducer_name,
            a.state_->as_jit()
        ));
    }

    Operation argument_axis_reduce(const Operation& a,
                                   const std::string& reducer_name,
                                   const int& axis) {
        int ndim = a.ndim();
        if (ndim == 0) return Operation(0);
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
        return Operation(std::make_shared<ArgumentAxisReducerOperationState>(
            reducer_name,
            res
        ));
    }
}  // namespace op2
