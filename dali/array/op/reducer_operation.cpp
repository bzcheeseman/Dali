#include "reducer_operation.h"

#include <algorithm>

#include "dali/array/op/elementwise_operation.h"
#include "dali/array/jit/jit.h"
#include "dali/array/jit/all_reduce_kernel_utils.h"

#include "dali/utils/hash_utils.h"
#include "dali/utils/make_message.h"

namespace op {
namespace jit {
struct Reducer : public JITNode {
    const std::string functor_name_;

    // MUST IMPLEMENT
    virtual std::string kernel_name() const = 0;

    // DO NOT REIMPLEMENT
    Reducer(const std::string& functor_name,
            const Array& argument,
            const std::vector<int>& output_shape,
            DType dtype) :
        JITNode(output_shape, dtype, {argument}), functor_name_(functor_name) {
    }

    virtual void compilation_parameters(utils::Hasher& hasher) const override {
        hasher.add(functor_name_);
    }

    virtual std::string get_call_code_nd(const SymbolTable& symbol_table,
                                         memory::DeviceT device_type) const override {
        return utils::make_message(
            kernel_name(), std::max(1, arguments_[0].ndim()),
            "d<", functor_name_, ", " , dtype_to_cpp_name(dtype_), ">(",
            op::jit::get_call_code_nd(arguments_[0], symbol_table, device_type), ")");
    }
};  // struct Reducer

struct AllReduce : public Reducer {
    AllReduce(const std::string& functor_name,
              const Array& argument, DType dtype) :
        Reducer(functor_name, argument, {}, dtype) {
    }

    virtual int min_computation_rank() const override {
        return 1;
    }

    virtual bool can_jit_right_fit_inputs() const override {
        return op::jit::min_computation_rank(arguments_[0]) < arguments_[0].ndim();
    }

    virtual expression_ptr jit_right_fit_ndim(int ndim) const override {
        // collapse child to become as collapsed as possible
        return std::make_shared<AllReduce>(
            functor_name_,
            op::jit::jit_right_fit_ndim(
                arguments_[0],
                op::jit::min_computation_rank(arguments_[0])
            ),
            dtype_
        );
    }

    virtual std::string name() const override {
        return utils::make_message("all_reduce<", functor_name_, ">");
    }

    virtual bool is_axis_collapsible_with_axis_minus_one(int dim) const override {
        return true;
    }

    virtual void prefix_code(memory::DeviceT device_type, insert_t insert) const override {
        create_all_reduce_kernel_caller(std::max(1, arguments_[0].ndim()), insert);
    }

    virtual std::string kernel_name() const override {
        return "all_reduce_kernel_";
    }

    virtual expression_ptr copy() const override {
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
    AxisReduce(const std::string& functor_name,
               const Array& argument,
               DType dtype) : Reducer(functor_name, argument,
                                      axis_reducer_shape(argument),
                                      dtype) {}

    virtual int min_computation_rank() const override {
        return std::max(op::jit::min_computation_rank(arguments_[0]) - 1, 1);
    }

    virtual std::string name() const override {
        return utils::make_message("axis_reduce<", functor_name_, ">");
    }

    virtual bool is_axis_collapsible_with_axis_minus_one(int axis) const override {
        return arguments_[0].is_axis_collapsible_with_axis_minus_one(axis - 1);
    }

    virtual expression_ptr collapse_axis_with_axis_minus_one(int axis, const Array* owner) const override {
        return std::make_shared<AxisReduce>(
            functor_name_,
            arguments_[0].collapse_axis_with_axis_minus_one(axis - 1),
            dtype_
        );
    }

    virtual expression_ptr dimshuffle(const std::vector<int>& permutation, const Array* owner) const override {
        auto new_permutation = permutation;
        // add last dim of tensor with rank (permutation.size() + 1)
        new_permutation.emplace_back(permutation.size());
        return std::make_shared<AxisReduce>(
            functor_name_,
            arguments_[0].dimshuffle(new_permutation),
            dtype_
        );
    }

    virtual expression_ptr jit_right_fit_ndim(int ndim) const override {
        return std::make_shared<AxisReduce>(
            functor_name_,
            op::jit::jit_right_fit_ndim(arguments_[0], ndim + 1),
            dtype_);
    }

    virtual void prefix_code(memory::DeviceT device_type, insert_t insert) const override {
        create_axis_reduce_kernel_caller(std::max(1, arguments_[0].ndim()), insert);
    }

    virtual std::string kernel_name() const override {
        return "axis_reduce_kernel_";
    }

    virtual expression_ptr copy() const override {
        return std::make_shared<AxisReduce>(
            functor_name_, arguments_[0], dtype_
        );
    }
};

struct ArgumentAllReduce : public AllReduce {
    using AllReduce::AllReduce;
    virtual std::string name() const override {
        return utils::make_message(
            "argument_all_reduce<", functor_name_, ">"
        );
    }

    virtual void prefix_code(memory::DeviceT device_type, insert_t insert) const override {
        create_argument_all_reduce_kernel_caller(std::max(1, arguments_[0].ndim()), insert);
    }

    virtual std::string kernel_name() const override {
        return "argument_all_reduce_kernel_";
    }

    virtual expression_ptr jit_right_fit_ndim(int ndim) const override {
        // collapse child to become as collapsed as possible
        return std::make_shared<ArgumentAllReduce>(
            functor_name_,
            op::jit::jit_right_fit_ndim(
                arguments_[0],
                op::jit::min_computation_rank(arguments_[0])
            ),
            dtype_
        );
    }

    virtual expression_ptr copy() const override {
        return std::make_shared<ArgumentAllReduce>(
            functor_name_, arguments_[0], dtype_
        );
    }
};

struct ArgumentAxisReduce : public AxisReduce {
    using AxisReduce::AxisReduce;

    virtual std::string name() const override {
        return utils::make_message(
            "argument_axis_reduce<", functor_name_, ">"
        );
    }

    virtual void prefix_code(memory::DeviceT device_type, insert_t insert) const override {
        create_argument_axis_reduce_kernel_caller(std::max(1, arguments_[0].ndim()), insert);
    }

    virtual expression_ptr collapse_axis_with_axis_minus_one(int axis, const Array* owner) const override {
        return std::make_shared<ArgumentAxisReduce>(
            functor_name_,
            arguments_[0].collapse_axis_with_axis_minus_one(axis - 1),
            DTYPE_INT32
        );
    }

    virtual expression_ptr dimshuffle(const std::vector<int>& permutation, const Array* owner) const override {
        auto new_permutation = permutation;
        // add last dim of tensor with rank (permutation.size() + 1)
        new_permutation.emplace_back(permutation.size());
        return std::make_shared<ArgumentAxisReduce>(
            functor_name_,
            arguments_[0].transpose(new_permutation),
            DTYPE_INT32
        );
    }

    virtual std::string kernel_name() const override {
        return "argument_axis_reduce_kernel_";
    }
};

struct WarpAxisReduce : public AxisReduce {
    using AxisReduce::AxisReduce;

    virtual expression_ptr copy() const override {
        return std::make_shared<WarpAxisReduce>(
            functor_name_, arguments_[0], dtype_
        );
    }

    virtual void prefix_code(memory::DeviceT device_type, insert_t insert) const override {
        create_warp_axis_reduce_kernel_caller(std::max(1, arguments_[0].ndim()), insert);
    }

    virtual PARALLELISM_T parallelism_type() const override {
        return INDEPENDENT_BLOCK;
    }

    virtual std::string kernel_name() const override {
        return "warp_axis_reduce_kernel_";
    }
};

struct WarpAllReduce : public AllReduce {
    using AllReduce::AllReduce;

    virtual expression_ptr copy() const override {
        return std::make_shared<WarpAllReduce>(
            functor_name_, arguments_[0], dtype_
        );
    }

    virtual void prefix_code(memory::DeviceT device_type, insert_t insert) const override {
        create_warp_all_reduce_kernel_caller(std::max(1, arguments_[0].ndim()), insert);
    }

    virtual PARALLELISM_T parallelism_type() const override {
        return INDEPENDENT_BLOCK;
    }

    virtual std::string kernel_name() const override {
        return "warp_all_reduce_kernel_";
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

    // TODO(jonathan) ensure word size checking happens at compile time
    std::string thread_sum(std::string dtype, int word_length) {
        return utils::make_message(
            "template<typename T, int ndim>\n"
            "inline __device__ ", dtype, " thread_sum(const ArrayView<", dtype, ", ndim>& input, Shape<ndim> query, int start, int stride) {\n"
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


    std::string thread_sum_generic() {
        return utils::make_message("template<typename T, typename C1, int ndim>\n"
           "inline __device__ T thread_sum(const C1& input, Shape<ndim> query, int start, int stride) {\n"
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

    std::string thread_sum_all(std::string dtype, int word_length) {
        return utils::make_message(
            "template<typename T, int ndim>\n"
            "inline __device__ ", dtype, " thread_sum_all(const ArrayView<", dtype, ", ndim>& input, int start, int stride) {\n"
            "    ", dtype, " sum = 0;\n"
            "    ", dtype, word_length, "* ptr = (", dtype, word_length, "*) &input[Shape<ndim>(0)];\n"
            "    int cols_div_word_length = input.shape()[ndim-1] / ", word_length, ";\n"
            "    int numel = input.shape().numel();\n"
            "    int i;\n"
            "    if (cols_div_word_length * ", word_length, " == numel) {\n"
            "        for(i = start;\n"
            "            i < cols_div_word_length;\n"
            "            i += stride) {\n"
            "            ", dtype, word_length, " in = ptr[i];\n"
            "            sum += ", word_length_to_sum(word_length), ";\n"
            "        }\n"
            "    } else {\n"
            "        for(i = start;\n"
            "            i < numel;\n"
            "            i += stride) {\n"
            "            sum += input[index_to_dim(i, input.shape())];\n"
            "        }\n"
            "    }\n"
            "    return sum;\n"
            "}\n");
    }

    std::string thread_sum_all_generic() {
        return utils::make_message("template<typename T, typename C1>\n"
           "inline __device__ T thread_sum_all(const C1& input, int start, int stride) {\n"
           "    T sum = 0;\n"
           "    int i;\n"
           "    auto shape = input.shape();\n"
           "    int numel = shape.numel();\n"
           "    for(i = start;\n"
           "        i < numel;\n"
           "        i += stride) {\n"
           "        sum += input[index_to_dim(i, shape)];\n"
           "    }\n"
           "    return sum;\n"
           "}\n");
    }

    std::string reduce_sum_tile_shfl() {
        return "template <int tile_sz, typename T>\n"
               "__device__ T reduce_sum_tile_shfl(cooperative_groups::thread_block_tile<tile_sz> g, T val) {\n"
               "    int lane = g.thread_rank();\n"
               "    // Each iteration halves the number of active threads\n"
               "    // Each thread adds its partial sum[i] to sum[lane+i]\n"
               "    for (int i = g.size() / 2; i > 0; i /= 2) {\n"
               "        val += g.shfl_down(val, i);\n"
               "    }\n"
               "    return val; // note: only thread 0 will return full sum\n"
               "}\n";
    }
}

struct ShflDownWarpAxisSum : public AxisReduce {
    using AxisReduce::AxisReduce;

    virtual expression_ptr copy() const override {
        return std::make_shared<ShflDownWarpAxisSum>(
            functor_name_, arguments_[0], dtype_
        );
    }

    virtual void prefix_code(memory::DeviceT device_type, insert_t insert) const override {
        // tunable variable:
        int tile_sz = 16;
        int rank = std::max(ndim(), 1);
        std::string clsname = "ShflDownWarpAxisSum";
        insert("#include <cooperative_groups.h>\n");
        insert(thread_sum_generic());
        // vector load versions of threadsum:
        insert(thread_sum("int", 4));
        insert(thread_sum("float", 4));
        insert(thread_sum("double", 2));
        insert(reduce_sum_tile_shfl());
        insert(utils::make_message(
            "template<typename Reducer, typename Type, typename C1>\n"
            "struct ", clsname, rank, " {\n"
            "    C1 arg_;\n"
            "    static const int ndim = C1::ndim - 1;\n"
            "    typedef Type T;\n"
            "    XINLINE Shape<ndim> shape() const {\n"
            "        return arg_.shape().template axis_reduced_shape<0, ndim>();\n"
            "    }\n"
            "    XINLINE ", clsname, rank, "(C1 arg) : arg_(arg) {}\n"
            "    inline __device__ T operator[](const Shape<ndim>& input_query) const {\n"
            "        __shared__ T sum;\n"
            "        sum = 0;\n"
            "        Shape<ndim + 1> query = input_query.expand_dims(ndim);\n"
            "        query[ndim] = 0;\n"
            "        T my_sum = thread_sum<T>(arg_, query, threadIdx.x, blockDim.x);\n"
            "        auto tile = cooperative_groups::tiled_partition<", tile_sz, ">(\n"
            "            cooperative_groups::this_thread_block());\n"
            "        T tile_sum = reduce_sum_tile_shfl<", tile_sz, ">(tile, my_sum);\n"
            "        if (tile.thread_rank() == 0) atomicAdd(&sum, tile_sum);\n"
            "        __syncthreads();\n"
            "        return sum;\n"
            "    }\n"
            "};\n"));
        insert(utils::make_message(
            "template<typename Reducer, typename Type, typename C1>\n"
            "XINLINE ", clsname, rank, "<Reducer, Type, C1> ", kernel_name(), rank + 1, "d(\n"
            "        C1 arg) {\n"
            "    return ", clsname, rank, "<Reducer, Type, C1>(arg);\n"
            "}\n"));
    }

    virtual PARALLELISM_T parallelism_type() const override {
        return INDEPENDENT_BLOCK;
    }

    virtual std::string kernel_name() const override {
        return "shfl_down_warp_axis_sum";
    }
};


struct ShflDownWarpAllSum : public AllReduce {
    using AllReduce::AllReduce;

    virtual expression_ptr copy() const override {
        return std::make_shared<ShflDownWarpAllSum>(
            functor_name_, arguments_[0], dtype_
        );
    }

    virtual void prefix_code(memory::DeviceT device_type, insert_t insert) const override {
        // tunable variable:
        int tile_sz = 16;
        int rank = std::max(ndim(), 1);
        int arg_rank = std::max(1, arguments_[0].ndim());
        std::string clsname = "ShflDownWarpAllSum";

        insert("#include <cooperative_groups.h>\n");
        insert(thread_sum_all_generic());
        insert(thread_sum_all("int", 4));
        insert(thread_sum_all("float", 4));
        insert(thread_sum_all("double", 2));
        insert(reduce_sum_tile_shfl());
        insert(utils::make_message(
            "template<typename Reducer, typename Type, typename C1>\n"
            "struct ", clsname, arg_rank, " {\n"
            "    C1 arg_;\n"
            "    static const int ndim = 1;\n"
            "    typedef Type T;\n"
            "    XINLINE Shape<ndim> shape() const {\n"
            "        return Shape<ndim>(1);\n"
            "    }\n"
            "    XINLINE ", clsname, arg_rank, "(C1 arg) : arg_(arg) {}\n"
            "    inline __device__ T operator[](const Shape<1>&) const {\n"
            "        __shared__ T sum;\n"
            "        sum = 0;\n"
            "        T my_sum = thread_sum_all<T>(arg_, threadIdx.x, blockDim.x);\n"
            "        auto tile = cooperative_groups::tiled_partition<", tile_sz, ">(\n"
            "            cooperative_groups::this_thread_block());\n"
            "        T tile_sum = reduce_sum_tile_shfl<", tile_sz, ">(tile, my_sum);\n"
            "        if (tile.thread_rank() == 0) atomicAdd(&sum, tile_sum);\n"
            "        __syncthreads();\n"
            "        return sum;\n"
            "    }\n"
            "};\n"));
        insert(utils::make_message(
            "template<typename Reducer, typename Type, typename C1>\n"
            "XINLINE ", clsname, arg_rank, "<Reducer, Type, C1> ", kernel_name(), arg_rank, "d(\n"
            "        C1 arg) {\n"
            "    return ", clsname, arg_rank, "<Reducer, Type, C1>(arg);\n"
            "}\n"));
    }

    virtual PARALLELISM_T parallelism_type() const override {
        return INDEPENDENT_BLOCK;
    }

    virtual std::string kernel_name() const override {
        return "shfl_down_warp_all_sum";
    }
};

namespace {
    AxisReduce* static_as_axis_reduce(const Array& array) {
        return static_cast<AxisReduce*>(array.expression().get());
    }
    AllReduce* static_as_all_reduce(const Array& array) {
        return static_cast<AllReduce*>(array.expression().get());
    }
    // convert a reduction to a warp reduction if
    // the warp dimension is still available
    // & the device is a GPU && the op is a reduction.
    bool can_transform_to_blocked_reducer(const Array& array,
                                          memory::DeviceT device_type) {
        return (device_type == memory::DEVICE_T_GPU &&
                typeid(*array.expression().get()).name() == typeid(AxisReduce).name() &&
                static_as_jit_node(array)->parallelism_type() == INDEPENDENT_BLOCK_WARP);
    }

    bool can_transform_to_blocked_allreducer(const Array& array,
                                             memory::DeviceT device_type) {
        return (device_type == memory::DEVICE_T_GPU &&
                typeid(*array.expression().get()).name() == typeid(AllReduce).name() &&
                static_as_jit_node(array)->parallelism_type() == INDEPENDENT_BLOCK_WARP);
    }

    bool can_transform_to_shfl_down_sum(const Array& array,
                                        memory::DeviceT device_type) {
        return (can_transform_to_blocked_reducer(array, device_type) &&
                static_as_axis_reduce(array)->functor_name_ == "reducers::sum" &&
                DALI_CUDA_VERSION >= 9.0);
    }

    bool can_transform_to_shfl_down_allsum(const Array& array,
                                        memory::DeviceT device_type) {
        return (can_transform_to_blocked_allreducer(array, device_type) &&
                static_as_all_reduce(array)->functor_name_ == "reducers::sum" &&
                DALI_CUDA_VERSION >= 9.0);
    }

    int register_axis_blocked = register_jit_optimization(
        1,
        can_transform_to_blocked_reducer,
        [](const Array& array) {
            auto r = static_as_axis_reduce(array);;
            return Array(std::make_shared<WarpAxisReduce>(
                r->functor_name_, r->arguments_[0], r->dtype_));
        },
        "warp_axis_reduce");
    int register_shfl_down_axis = register_jit_optimization(
        0,
        can_transform_to_shfl_down_sum,
        [](const Array& array) {
            auto r = static_as_axis_reduce(array);
            return Array(std::make_shared<ShflDownWarpAxisSum>(
                r->functor_name_, r->arguments_[0], r->dtype_));},
        "shfl_down_axis_sum");
    int register_all_blocked = register_jit_optimization(
        1,
        can_transform_to_blocked_allreducer,
        [](const Array& array) {
            auto r = static_as_all_reduce(array);;
            return Array(std::make_shared<WarpAllReduce>(
                r->functor_name_, r->arguments_[0], r->dtype_));},
        "warp_all_reduce");
    int register_all_reduce_shfl_down = register_jit_optimization(
        0,
        can_transform_to_shfl_down_allsum,
        [](const Array& array) {
            auto r = static_as_all_reduce(array);
            return Array(std::make_shared<ShflDownWarpAllSum>(
                r->functor_name_, r->arguments_[0], r->dtype_));},
        "shfl_down_all_sum");
}
} // namespace jit

Array all_reduce(
        const Array& a,
        const std::string& reducer_name) {
    return Array(std::make_shared<op::jit::AllReduce>(
        reducer_name, a, a.dtype()));
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
                " ndim=", ndim, ", input.shape = ", a.shape(), ")."));
    }
    // now look to see what kind of a reduction this is:
    std::vector<bool> reduced_dims(ndim, false);
    std::sort(normalized_axes.begin(), normalized_axes.end());
    for (auto& axis : normalized_axes) {
        ASSERT2(!reduced_dims[axis], utils::make_message("axis_reduce "
            "received duplicate axes to operate on (axis=", axis,
            " axes=", axes, ")."));
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
        reducer_name, a, a.dtype()));
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
            " ndim=", ndim, ")."));
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
        reducer_name, res, DTYPE_INT32));
}
}  // namespace op
