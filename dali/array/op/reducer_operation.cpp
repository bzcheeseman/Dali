#include "reducer_operation.h"
#include "dali/array/op/elementwise_operation.h"
#include "dali/array/jit/jit_runner.h"
#include "dali/array/jit/all_reduce_kernel_utils.h"

#include "dali/utils/hash_utils.h"
#include "dali/utils/make_message.h"

namespace op {
namespace jit {
struct ReducerExpression : public JITNode {
    Array argument_;
    const std::string functor_name_;

    // MUST IMPLEMENT
    virtual hash_t optype_hash() const = 0;
    virtual std::string kernel_name() const = 0;

    // DO NOT REIMPLEMENT
    ReducerExpression(const std::string& functor_name,
                      const Array& argument,
                      const std::vector<int>& output_shape,
                      DType dtype, int min_computation_rank) :
        JITNode(min_computation_rank, output_shape, dtype),
        functor_name_(functor_name),
        argument_(argument) {
    }

    virtual std::vector<Array> arguments() const {
        return {argument_};
    }

    virtual std::string get_call_code_nd(const SymbolTable& symbol_table,
                                         const node_to_info_t& node_to_info,
                                         memory::DeviceT device_type) const {
        int all_reduce_comp_rank = node_to_info.at(argument_.expression().get()).computation_rank;
        return utils::make_message(
            kernel_name(), all_reduce_comp_rank,
            "d<", functor_name_, ", " , dtype_to_cpp_name(dtype_), ">(",
            op::jit::get_call_code_nd(argument_, symbol_table, node_to_info, device_type), ")");
    }
};  // struct ReducerExpression

struct AllReducerExpression : public ReducerExpression {
    static const hash_t optype_hash_cache_;

    virtual hash_t optype_hash() const {
        return optype_hash_cache_;
    }

    AllReducerExpression(const std::string& functor_name, const Array& argument, DType dtype);

    virtual std::string name() const {
        return utils::make_message(
            "all_reduce<", functor_name_, ">"
        );
    }

    virtual void compute_node_compilation_info(int desired_computation_rank,
                                               const std::vector<int>& desired_computation_shape,
                                               SymbolTable& symbol_table,
                                               node_to_info_t* node_to_info) const;

    virtual bool is_axis_collapsible_with_axis_minus_one(int dim) const {
        return true;
    }

    virtual std::string prefix_code(const node_to_info_t& node_to_info, memory::DeviceT device_type) const;

    virtual std::string kernel_name() const {
        return "all_reduce_kernel_";
    }

    virtual expression_ptr copy() const {
        return std::make_shared<AllReducerExpression>(
            functor_name_, argument_, argument_.dtype()
        );
    }
};

struct AxisReducerExpression : public ReducerExpression {
    static const hash_t optype_hash_cache_;

    virtual hash_t optype_hash() const {
        return optype_hash_cache_;
    }

    AxisReducerExpression(const std::string& functor_name, const Array& argument, DType dtype);

    virtual std::string name() const {
        return utils::make_message(
            "axis_reduce<", functor_name_, ">"
        );
    }

    virtual void compute_node_compilation_info(
        int desired_computation_rank,
        const std::vector<int>& desired_computation_shape,
        SymbolTable& symbol_table,
        node_to_info_t* node_to_info) const;

    virtual bool is_axis_collapsible_with_axis_minus_one(int dim) const;

    virtual expression_ptr collapse_axis_with_axis_minus_one(int dim) const;

    virtual expression_ptr transpose(const std::vector<int>& permutation) const;

    virtual std::string prefix_code(const node_to_info_t& node_to_info, memory::DeviceT device_type) const;

    virtual std::string kernel_name() const {
        return "axis_reduce_kernel_";
    }

    virtual expression_ptr copy() const {
        return std::make_shared<AxisReducerExpression>(
            functor_name_, argument_, argument_.dtype()
        );
    }
};

struct ArgumentAllReducerExpression : public AllReducerExpression {
    static const hash_t optype_hash_cache_;

    using AllReducerExpression::AllReducerExpression;

    virtual hash_t optype_hash() const {
        return optype_hash_cache_;
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

struct ArgumentAxisReducerExpression : public AxisReducerExpression {
    static const hash_t optype_hash_cache_;

    using AxisReducerExpression::AxisReducerExpression;

    virtual hash_t optype_hash() const {
        return optype_hash_cache_;
    }

    virtual std::string name() const {
        return utils::make_message(
            "argument_axis_reduce<", functor_name_, ">"
        );
    }

    virtual std::string prefix_code(const node_to_info_t& node_to_info, memory::DeviceT device_type) const;

    virtual expression_ptr collapse_axis_with_axis_minus_one(int axis) const;

    virtual expression_ptr transpose(const std::vector<int>& permutation) const;

    virtual std::string kernel_name() const {
        return "argument_axis_reduce_kernel_";
    }
};


///////////////////////////////////////////////////////////////////////////////
//                    ALL REDUCER OPERATION STATE                            //
///////////////////////////////////////////////////////////////////////////////


const hash_t AxisReducerExpression::optype_hash_cache_ = std::hash<std::string>()(
    typeid(AxisReducerExpression).name());

AllReducerExpression::AllReducerExpression(
        const std::string& functor_name,
        const Array& argument,
        DType dtype) :
    ReducerExpression(functor_name, argument, {}, dtype, 1) {
}

void AllReducerExpression::compute_node_compilation_info(
        int desired_computation_rank,
        const std::vector<int>& desired_computation_shape,
        SymbolTable& symbol_table,
        node_to_info_t* node_to_info) const {
    (*node_to_info)[this].computation_rank = desired_computation_rank;
    op::jit::compute_node_compilation_info(argument_,
                                           min_computation_rank(argument_),
                                           argument_.shape(),
                                           symbol_table,
                                           node_to_info);
    (*node_to_info)[this].hash = utils::Hasher().add(optype_hash())
                                                .add(desired_computation_rank)
                                                .add(functor_name_)
                                                .add(node_to_info->at(argument_.expression().get()).hash)
                                                .value();
}

std::string AllReducerExpression::prefix_code(
        const node_to_info_t& node_to_info,
        memory::DeviceT device_type) const {
    return create_all_reduce_kernel_caller(
        node_to_info.at(argument_.expression().get()).computation_rank,
        node_to_info.at(this).computation_rank
    );
}

const hash_t AllReducerExpression::optype_hash_cache_ = std::hash<std::string>()(
    typeid(AllReducerExpression).name());


/////////////////////////////////////////////////////////////////////////
//                    AXIS REDUCER OPERATION                           //
/////////////////////////////////////////////////////////////////////////

std::vector<int> axis_reducer_shape(const Array& a) {
    auto result = a.shape();
    result.pop_back();
    return result;
}

AxisReducerExpression::AxisReducerExpression(
        const std::string& functor_name,
        const Array& argument,
        DType dtype) :
    ReducerExpression(functor_name, argument,
        axis_reducer_shape(argument),
        dtype,
        std::max(op::jit::min_computation_rank(argument) - 1, 1)) {
}

void AxisReducerExpression::compute_node_compilation_info(
        int desired_computation_rank,
        const std::vector<int>& desired_computation_shape,
        SymbolTable& symbol_table,
        node_to_info_t* node_to_info) const {
    (*node_to_info)[this].computation_rank = desired_computation_rank;
    auto desired_argument_shape = desired_computation_shape;
    if (argument_.ndim() > 0) {
        desired_argument_shape.emplace_back(argument_.shape().back());
    } else {
        desired_argument_shape.emplace_back(1);
    }
    op::jit::compute_node_compilation_info(argument_,
                                           desired_computation_rank + 1,
                                           desired_argument_shape,
                                           symbol_table,
                                           node_to_info);
    (*node_to_info)[this].hash = utils::Hasher().add(optype_hash())
                                                .add(desired_computation_rank)
                                                .add(functor_name_)
                                                .add(node_to_info->at(argument_.expression().get()).hash)
                                                .value();
}

bool AxisReducerExpression::is_axis_collapsible_with_axis_minus_one(int dim) const {
    return argument_.is_axis_collapsible_with_axis_minus_one(dim - 1);
}

expression_ptr AxisReducerExpression::collapse_axis_with_axis_minus_one(int axis) const {
    return std::make_shared<AxisReducerExpression>(
        functor_name_,
        argument_.collapse_axis_with_axis_minus_one(axis - 1),
        argument_.dtype()
    );
}

expression_ptr AxisReducerExpression::transpose(
        const std::vector<int>& permutation) const {
    auto new_permutation = permutation;
    // add last dim of tensor with rank (permutation.size() + 1)
    new_permutation.emplace_back(permutation.size());
    return std::make_shared<AxisReducerExpression>(
        functor_name_,
        argument_.transpose(new_permutation),
        argument_.dtype()
    );
}

std::string AxisReducerExpression::prefix_code(
        const node_to_info_t& node_to_info,
        memory::DeviceT device_type) const {
    return create_axis_reduce_kernel_caller(node_to_info.at(argument_.expression().get()).computation_rank);
}


///////////////////////////////////////////////////////////////////////////////
//                ARGUMENT ALL REDUCER OPERATION STATE                       //
///////////////////////////////////////////////////////////////////////////////

std::string ArgumentAllReducerExpression::prefix_code(
        const node_to_info_t& node_to_info,
        memory::DeviceT device_type) const {
    return create_argument_all_reduce_kernel_caller(
        node_to_info.at(argument_.expression().get()).computation_rank,
        node_to_info.at(this).computation_rank
    );
}

const hash_t ArgumentAllReducerExpression::optype_hash_cache_ = std::hash<std::string>()(
    typeid(ArgumentAllReducerExpression).name());


///////////////////////////////////////////////////////////////////////////////
//         ARGUMENT AXIS REDUCER OPERATION STATE                             //
///////////////////////////////////////////////////////////////////////////////

const hash_t ArgumentAxisReducerExpression::optype_hash_cache_ = std::hash<std::string>()(
    typeid(ArgumentAxisReducerExpression).name());

std::string ArgumentAxisReducerExpression::prefix_code(
        const node_to_info_t& node_to_info,
        memory::DeviceT device_type) const {
    return create_argument_axis_reduce_kernel_caller(
        node_to_info.at(argument_.expression().get()).computation_rank);
}

expression_ptr ArgumentAxisReducerExpression::collapse_axis_with_axis_minus_one(int axis) const {
    return std::make_shared<ArgumentAxisReducerExpression>(
        functor_name_,
        argument_.collapse_axis_with_axis_minus_one(axis - 1),
        DTYPE_INT32
    );
}

expression_ptr ArgumentAxisReducerExpression::transpose(const std::vector<int>& permutation) const {
    auto new_permutation = permutation;
    // add last dim of tensor with rank (permutation.size() + 1)
    new_permutation.emplace_back(permutation.size());

    return std::make_shared<ArgumentAxisReducerExpression>(
        functor_name_,
        argument_.transpose(new_permutation),
        DTYPE_INT32
    );
}
}  // namespace jit

Array all_reduce(
        const Array& a,
        const std::string& reducer_name) {
    return Array(std::make_shared<op::jit::AllReducerExpression>(
        reducer_name,
        a,
        a.dtype()
    ));
}

Array axis_reduce(
        const Array& a,
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
                " ndim=", ndim, ", input.shape = ", a.shape(), ")."
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
    auto res = a;

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
        res = res.transpose(new_axes_order);
    }
    int num_low_axes_to_reduce = normalized_axes.size();
    if (num_low_axes_to_reduce > 0) {
        int axes_used_up = 0;
        int collapsed_ndim = ndim - 1;
        for (int axes_used_up = 0; axes_used_up < num_low_axes_to_reduce; ++axes_used_up) {
            if (num_low_axes_to_reduce - axes_used_up == 1  || !res.is_axis_collapsible_with_axis_minus_one(collapsed_ndim)) {
                res = Array(std::make_shared<op::jit::AxisReducerExpression>(
                    reducer_name,
                    res,
                    res.dtype()
                ));
            } else {
                res = res.collapse_axis_with_axis_minus_one(collapsed_ndim);
            }
            --collapsed_ndim;
        }
    }
    return res;
}

Array argument_all_reduce(const Array& a, const std::string& reducer_name) {
    return Array(std::make_shared<op::jit::ArgumentAllReducerExpression>(
        reducer_name,
        a,
        a.dtype()
    ));
}

Array argument_axis_reduce(const Array& a, const std::string& reducer_name, const int& axis) {
    int ndim = a.ndim();
    if (ndim == 0) return Array(0);
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

    auto res = a;
    if (normalized_axis != ndim - 1) {
        std::vector<int> axes;
        for (int i = 0; i < ndim; i++) {
            axes.emplace_back(i);
        }
        axes[axes.size() - 1] = normalized_axis;
        axes[normalized_axis] = axes.size() - 1;
        res = res.transpose(axes);
    }
    return Array(std::make_shared<op::jit::ArgumentAxisReducerExpression>(
        reducer_name,
        res,
        DTYPE_INT32
    ));
}
}  // namespace op
