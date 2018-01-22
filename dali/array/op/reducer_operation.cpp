#include "reducer_operation.h"

#include <algorithm>

#include "dali/array/op/elementwise_operation.h"
#include "dali/array/jit/jit_runner.h"
#include "dali/array/jit/all_reduce_kernel_utils.h"

#include "dali/utils/hash_utils.h"
#include "dali/utils/make_message.h"

namespace op {
namespace jit {
struct Reducer : public JITNode {
    const std::string functor_name_;

    // MUST IMPLEMENT
    virtual hash_t optype_hash() const = 0;
    virtual std::string kernel_name() const = 0;

    // DO NOT REIMPLEMENT
    Reducer(const std::string& functor_name,
            const Array& argument,
            const std::vector<int>& output_shape,
            DType dtype, int min_computation_rank) :
        JITNode(min_computation_rank, output_shape, dtype, {argument}),
        functor_name_(functor_name) {
    }

    virtual std::string get_call_code_nd(const SymbolTable& symbol_table,
                                         const node_to_info_t& node_to_info,
                                         memory::DeviceT device_type) const {
        int all_reduce_comp_rank = node_to_info.at(arguments_[0].expression().get()).computation_rank;
        return utils::make_message(
            kernel_name(), all_reduce_comp_rank,
            "d<", functor_name_, ", " , dtype_to_cpp_name(dtype_), ">(",
            op::jit::get_call_code_nd(arguments_[0], symbol_table, node_to_info, device_type), ")");
    }
};  // struct Reducer

struct AllReduce : public Reducer {
    static const hash_t optype_hash_cache_;

    virtual hash_t optype_hash() const {
        return optype_hash_cache_;
    }

    AllReduce(const std::string& functor_name,
                         const Array& argument, DType dtype) :
        Reducer(functor_name, argument, {}, dtype, 1) {
    }

    virtual std::string name() const {
        return utils::make_message("all_reduce<", functor_name_, ">");
    }

    virtual void compute_node_compilation_info(
            int desired_computation_rank,
            const std::vector<int>& desired_computation_shape,
            SymbolTable& symbol_table,
            node_to_info_t& node_to_info) const {
        node_to_info[this].computation_rank = desired_computation_rank;
        op::jit::compute_node_compilation_info(arguments_[0],
                                               min_computation_rank(arguments_[0]),
                                               arguments_[0].shape(),
                                               symbol_table,
                                               node_to_info);
        node_to_info[this].hash = utils::Hasher().add(optype_hash())
                                                    .add(desired_computation_rank)
                                                    .add(functor_name_)
                                                    .add(node_to_info.at(arguments_[0].expression().get()).hash)
                                                    .value();
        node_to_info[this].data_hash = compute_node_data_hash(node_to_info);
    }

    virtual bool is_axis_collapsible_with_axis_minus_one(int dim) const {
        return true;
    }

    virtual std::string prefix_code(const node_to_info_t& node_to_info,
                                    memory::DeviceT device_type) const {
        return create_all_reduce_kernel_caller(
            node_to_info.at(arguments_[0].expression().get()).computation_rank,
            node_to_info.at(this).computation_rank
        );
    }

    virtual std::string kernel_name() const {
        return "all_reduce_kernel_";
    }

    virtual expression_ptr copy() const {
        return std::make_shared<AllReduce>(
            functor_name_, arguments_[0], dtype_
        );
    }
};

namespace {
    std::vector<int> axis_reducer_shape(const Array& a) {
        auto result = a.shape();
        result.pop_back();
        return result;
    }
}

struct AxisReduce : public Reducer {
    static const hash_t optype_hash_cache_;

    virtual hash_t optype_hash() const {
        return optype_hash_cache_;
    }

    AxisReduce(const std::string& functor_name,
               const Array& argument,
               DType dtype) : Reducer(functor_name, argument,
                                      axis_reducer_shape(argument),
                                      dtype,
                                      std::max(op::jit::min_computation_rank(argument) - 1, 1)) {}

    virtual std::string name() const {
        return utils::make_message("axis_reduce<", functor_name_, ">");
    }

    virtual void compute_node_compilation_info(
            int desired_computation_rank,
            const std::vector<int>& desired_computation_shape,
            SymbolTable& symbol_table,
            node_to_info_t& node_to_info) const {
        node_to_info[this].computation_rank = desired_computation_rank;
        auto desired_argument_shape = desired_computation_shape;
        if (arguments_[0].ndim() > 0) {
            desired_argument_shape.emplace_back(arguments_[0].shape().back());
        } else {
            desired_argument_shape.emplace_back(1);
        }
        op::jit::compute_node_compilation_info(arguments_[0],
                                               desired_computation_rank + 1,
                                               desired_argument_shape,
                                               symbol_table,
                                               node_to_info);
        node_to_info[this].hash = utils::Hasher().add(optype_hash())
                                                    .add(desired_computation_rank)
                                                    .add(functor_name_)
                                                    .add(node_to_info.at(arguments_[0].expression().get()).hash)
                                                    .value();
        node_to_info[this].data_hash = compute_node_data_hash(node_to_info);
    }

    virtual bool is_axis_collapsible_with_axis_minus_one(int axis) const {
        return arguments_[0].is_axis_collapsible_with_axis_minus_one(axis - 1);
    }

    virtual expression_ptr collapse_axis_with_axis_minus_one(int axis) const {
        return std::make_shared<AxisReduce>(
            functor_name_,
            arguments_[0].collapse_axis_with_axis_minus_one(axis - 1),
            dtype_
        );
    }

    virtual expression_ptr transpose(const std::vector<int>& permutation) const {
        auto new_permutation = permutation;
        // add last dim of tensor with rank (permutation.size() + 1)
        new_permutation.emplace_back(permutation.size());
        return std::make_shared<AxisReduce>(
            functor_name_,
            arguments_[0].transpose(new_permutation),
            dtype_
        );
    }

    virtual std::string prefix_code(const node_to_info_t& node_to_info,
                                    memory::DeviceT device_type) const {
        return create_axis_reduce_kernel_caller(
            node_to_info.at(arguments_[0].expression().get()).computation_rank);
    }

    virtual std::string kernel_name() const {
        return "axis_reduce_kernel_";
    }

    virtual expression_ptr copy() const {
        return std::make_shared<AxisReduce>(
            functor_name_, arguments_[0], dtype_
        );
    }
};

struct ArgumentAllReduce : public AllReduce {
    static const hash_t optype_hash_cache_;

    using AllReduce::AllReduce;

    virtual hash_t optype_hash() const {
        return optype_hash_cache_;
    }

    virtual std::string name() const {
        return utils::make_message(
            "argument_all_reduce<", functor_name_, ">"
        );
    }

    virtual std::string prefix_code(const node_to_info_t& node_to_info,
                                    memory::DeviceT device_type) const {
        return create_argument_all_reduce_kernel_caller(
            node_to_info.at(arguments_[0].expression().get()).computation_rank,
            node_to_info.at(this).computation_rank
        );
    }

    virtual std::string kernel_name() const {
        return "argument_all_reduce_kernel_";
    }
};

struct ArgumentAxisReduce : public AxisReduce {
    static const hash_t optype_hash_cache_;

    using AxisReduce::AxisReduce;

    virtual hash_t optype_hash() const {
        return optype_hash_cache_;
    }

    virtual std::string name() const {
        return utils::make_message(
            "argument_axis_reduce<", functor_name_, ">"
        );
    }

    virtual std::string prefix_code(const node_to_info_t& node_to_info,
                                    memory::DeviceT device_type) const {
        return create_argument_axis_reduce_kernel_caller(
            node_to_info.at(arguments_[0].expression().get()).computation_rank);
    }

    virtual expression_ptr collapse_axis_with_axis_minus_one(int axis) const {
        return std::make_shared<ArgumentAxisReduce>(
            functor_name_,
            arguments_[0].collapse_axis_with_axis_minus_one(axis - 1),
            DTYPE_INT32
        );
    }

    virtual expression_ptr transpose(const std::vector<int>& permutation) const {
        auto new_permutation = permutation;
        // add last dim of tensor with rank (permutation.size() + 1)
        new_permutation.emplace_back(permutation.size());
        return std::make_shared<ArgumentAxisReduce>(
            functor_name_,
            arguments_[0].transpose(new_permutation),
            DTYPE_INT32
        );
    }

    virtual std::string kernel_name() const {
        return "argument_axis_reduce_kernel_";
    }
};

struct WarpAxisReduce : public AxisReduce {
    static const hash_t optype_hash_cache_;
    using AxisReduce::AxisReduce;

    virtual hash_t optype_hash() const {
        return optype_hash_cache_;
    }

    virtual expression_ptr copy() const {
        return std::make_shared<WarpAxisReduce>(
            functor_name_, arguments_[0], dtype_
        );
    }

    virtual std::string prefix_code(const node_to_info_t& node_to_info,
                                    memory::DeviceT device_type) const {
        return create_warp_axis_reduce_kernel_caller(
            node_to_info.at(arguments_[0].expression().get()).computation_rank);
    }

    virtual PARALLELISM_T parallelism_type() const override {
        return INDEPENDENT_BLOCK;
    }

    virtual std::string kernel_name() const {
        return "warp_axis_reduce_kernel_";
    }
};

namespace {

    std::string word_length_to_sum(int word_length) {
        if (word_length == 1) {
            return "in.x";
        } else if (word_length == 2) {
            return "in.x + in.y";
        } else if (word_length == 3) {
            return "in.x + in.y + in.z";
        } else if (word_length == 4) {
            return "in.x + in.y + in.z + in.w";
        } else {
            ASSERT2(false, utils::make_message(
                "no know sum with word_length ", word_length, "."));
        }
    }

    std::string thread_sum(std::string dtype, int word_length, int ndim) {
        return utils::make_message(
            "template<typename T, int ndim>\n"
            "inline __device__ ", dtype, " thread_sum", ndim, "(const ArrayView<", dtype, ", ndim>& input, Shape<ndim> query, int start, int stride) {\n"
            "    ", dtype, " sum = 0;\n"
            "    ", dtype, word_length, "* ptr = (", dtype, word_length, "*) &input[query];\n"
            "    int cols_div_word_length = input.shape()[ndim-1] / ", word_length, ";\n"
            "    int& i = query[ndim - 1];\n"
            "    if (cols_div_word_length * ", word_length, " == input.shape()[ndim-1]) {\n"
            "        for(i = start;\n"
            "            i < cols_div_word_length;\n"
            "            i += stride) {\n"
            "            ", dtype, word_length, " in = ptr[i];\n"
            "            sum += ", word_length_to_sum(word_length), ";\n"
            "        }\n"
            "    } else {\n"
            "        for(i = start;\n"
            "            i < input.shape()[ndim-1];\n"
            "            i += stride) {\n"
            "            sum += input[query];\n"
            "        }\n"
            "    }\n"
            "    return sum;\n"
            "}\n");
    }

    std::string thread_sum_generic(int ndim) {
        return utils::make_message("template<typename T, typename C1, int ndim>\n"
           "inline __device__ T thread_sum", ndim, "(const C1& input, Shape<ndim> query, int start, int stride) {\n"
           "    T sum = 0;\n"
           "    int& i = query[ndim - 1];\n"
           "    for(i = start;\n"
           "        i < input.shape()[ndim-1];\n"
           "        i += stride) {\n"
           "        sum += input[query];\n"
           "    }\n"
           "    return sum;\n"
           "}\n");
    }
}

struct ShflDownWarpAxisSum : public AxisReduce {
    static const hash_t optype_hash_cache_;
    using AxisReduce::AxisReduce;

    virtual hash_t optype_hash() const {
        return optype_hash_cache_;
    }

    virtual expression_ptr copy() const {
        return std::make_shared<ShflDownWarpAxisSum>(
            functor_name_, arguments_[0], dtype_
        );
    }

    virtual std::string prefix_code(const node_to_info_t& node_to_info,
                                    memory::DeviceT device_type) const {
        // tunable variable:
        int tile_sz = 16;
        int ndim = node_to_info.at(this).computation_rank;
        return utils::make_message(
            "#include <cooperative_groups.h>\n",
            thread_sum_generic(ndim),
            // vector load versions of threadsum:
            thread_sum("int", 4, ndim),
            thread_sum("float", 4, ndim),
            thread_sum("double", 2, ndim),
            "template <int tile_sz, typename T>\n"
            "__device__ T reduce_sum_tile_shfl", ndim, "(cooperative_groups::thread_block_tile<tile_sz> g, T val) {\n"
            "    int lane = g.thread_rank();\n"
            "    // Each iteration halves the number of active threads\n"
            "    // Each thread adds its partial sum[i] to sum[lane+i]\n"
            "    for (int i = g.size() / 2; i > 0; i /= 2) {\n"
            "        val += g.shfl_down(val, i);\n"
            "    }\n"
            "    return val; // note: only thread 0 will return full sum\n"
            "}\n"
            "template<typename Reducer, typename Type, typename C1>\n"
            "struct ShflDownWarpAxisSum", ndim, " {\n"
            "    C1 arg_;\n"
            "    static const int ndim = ", ndim, ";\n"
            "    typedef Type T;\n"
            "    XINLINE Shape<ndim> shape() const {\n"
            "        return arg_.shape().template axis_reduced_shape<0, ndim>();\n"
            "    }\n"
            "    XINLINE ShflDownWarpAxisSum", ndim, "(C1 arg) : arg_(arg) {}\n"
            "    inline __device__ T operator[](const Shape<", ndim, ">& input_query) const {\n"
            "        __shared__ T sum;\n"
            "        sum = 0;\n"
            "        Shape<", ndim + 1, "> query = input_query.expand_dims(", ndim, ");\n"
            "        query[", ndim, "] = 0;\n"
            "        T my_sum = thread_sum", ndim, "<T>(arg_, query, threadIdx.x, blockDim.x);\n"
            "        auto tile = cooperative_groups::tiled_partition<", tile_sz, ">(\n"
            "            cooperative_groups::this_thread_block());\n"
            "        T tile_sum = reduce_sum_tile_shfl", ndim, "<", tile_sz, ">(tile, my_sum);\n"
            "        if (tile.thread_rank() == 0) atomicAdd(&sum, tile_sum);\n"
            "        __syncthreads();\n"
            "        return sum;\n"
            "    }\n"
            "};\n"
            "template<typename Reducer, typename Type, typename C1>\n"
            "XINLINE ShflDownWarpAxisSum", ndim, "<Reducer, Type, C1> shfl_down_warp_axis_sum", ndim + 1, "d(\n"
            "        C1 arg) {\n"
            "    return ShflDownWarpAxisSum", ndim, "<Reducer, Type, C1>(arg);\n"
            "}\n"
        );
    }

    virtual PARALLELISM_T parallelism_type() const override {
        return INDEPENDENT_BLOCK;
    }

    virtual std::string kernel_name() const {
        return "shfl_down_warp_axis_sum";
    }
};

const hash_t AxisReduce::optype_hash_cache_ = std::hash<std::string>()(
    typeid(AxisReduce).name());

const hash_t WarpAxisReduce::optype_hash_cache_ = std::hash<std::string>()(
    typeid(WarpAxisReduce).name());

const hash_t ShflDownWarpAxisSum::optype_hash_cache_ = std::hash<std::string>()(
    typeid(ShflDownWarpAxisSum).name());

const hash_t AllReduce::optype_hash_cache_ = std::hash<std::string>()(
    typeid(AllReduce).name());

const hash_t ArgumentAllReduce::optype_hash_cache_ = std::hash<std::string>()(
    typeid(ArgumentAllReduce).name());

const hash_t ArgumentAxisReduce::optype_hash_cache_ = std::hash<std::string>()(
    typeid(ArgumentAxisReduce).name());


namespace {
    AxisReduce* static_as_axis_reduce(const Array& array) {
        return static_cast<AxisReduce*>(array.expression().get());
    }
    // convert a reduction to a warp reduction if
    // the warp dimension is still available
    // & the device is a GPU && the op is a reduction.
    bool can_transform_to_blocked_reducer(const Array& array,
                                          memory::DeviceT device_type,
                                          const node_to_info_t& node_to_info) {
        return (device_type == memory::DEVICE_T_GPU &&
                typeid(*array.expression().get()).name() == typeid(AxisReduce).name() &&
                static_as_jit_node(array)->parallelism_type() == INDEPENDENT_BLOCK_WARP);
    }

    Array transform_to_blocked_reducer(const Array& array) {
        auto reducer = static_as_axis_reduce(array);;
        return Array(std::make_shared<WarpAxisReduce>(
            reducer->functor_name_,
            reducer->arguments_[0],
            reducer->dtype_
        ));
    }

    bool can_transform_to_shfl_down_sum(const Array& array,
                                        memory::DeviceT device_type,
                                        const node_to_info_t& node_to_info) {
        return (can_transform_to_blocked_reducer(array, device_type, node_to_info) &&
                static_as_axis_reduce(array)->functor_name_ == "reducers::sum" &&
                DALI_CUDA_VERSION >= 9.0);
    }

    Array transform_to_shfl_down_sum(const Array& array) {
        auto reducer = static_as_axis_reduce(array);
        return Array(std::make_shared<ShflDownWarpAxisSum>(
            reducer->functor_name_,
            reducer->arguments_[0],
            reducer->dtype_
        ));
    }
}

int register_axis_blocked = register_jit_optimization(
    1,
    can_transform_to_blocked_reducer,
    transform_to_blocked_reducer,
    "warp_axis_reduce");
int register_shfl_down_axis = register_jit_optimization(
    0,
    can_transform_to_shfl_down_sum,
    transform_to_shfl_down_sum,
    "shfl_down_axis_sum");

} // namespace jit

Array all_reduce(
        const Array& a,
        const std::string& reducer_name) {
    return Array(std::make_shared<op::jit::AllReduce>(
        reducer_name,
        a,
        a.dtype()
    ));
}

Array axis_reduce(
        const Array& a,
        const std::string& reducer_name,
        const std::vector<int>& axes,
        bool keepdims) {
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
                res = Array(std::make_shared<op::jit::AxisReduce>(
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
    if (keepdims) {
        for (auto& axis : normalized_axes) {
            res = res.expand_dims(axis);
        }
    }
    return res;
}

Array argument_all_reduce(const Array& a, const std::string& reducer_name) {
    return Array(std::make_shared<op::jit::ArgumentAllReduce>(
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
            " ndim=", ndim, ").")
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
    return Array(std::make_shared<op::jit::ArgumentAxisReduce>(
        reducer_name, res, DTYPE_INT32
    ));
}
}  // namespace op
