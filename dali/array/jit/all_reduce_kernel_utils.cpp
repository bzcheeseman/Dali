#include "elementwise_kernel_utils.h"
#include "dali/utils/make_message.h"
#include "dali/utils/print_utils.h"
#include "dali/array/jit/jit_utils.h"

namespace {
    std::string generate_reduce_kernel_class_signature(std::string name, int ndim, int nargs) {
        utils::MS stream;
        stream << name << ndim << "<Reducer, Type";
        for (int i = 0; i < nargs; i++) {
            stream << ", C" << (i+1);
        }
        stream << ">";
        return stream;
    }

    std::string generate_all_reduce_kernel_argument(int num) {
        return utils::make_message("arg_", num, "_view");
    }

    std::string generate_all_reduce_kernel_arguments(int nargs) {
        utils::MS stream;
        for (int i = 0; i < nargs; i++) {
            stream << generate_all_reduce_kernel_argument(i + 1);
            if (i + 1 != nargs) {
                stream << ", ";
            }
        }
        return stream;
    }

    std::string generate_reduce_kernel_template_code(int num) {
        utils::MS stream;
        stream << "template<typename Reducer, typename Type";
        for (int i = 0; i < num;i++) {
            stream << ", typename C" << (i+1);
        }
        stream << ">\n";
        return stream;
    }

    std::string generate_reduce_kernel_constructor_arguments(int nargs) {
        utils::MS stream;
        for (int i = 0; i < nargs; i++) {
            stream << "        C" << (i + 1) << " "
                   << generate_all_reduce_kernel_argument(i+1);
            if (i + 1 != nargs) {
                stream << ",\n";
            } else {
                stream << ")";
            }
        }
        return stream;
    }

    std::string generate_axis_reduce_kernel_constructor_arguments() {
        return utils::make_message("        "
            "const C1& ", generate_all_reduce_kernel_argument(1), ", "
            "const C2& ", generate_all_reduce_kernel_argument(2), ")"
        );
    }

    std::string generate_all_reduce_kernel_constructor(std::string name, int ndim) {
        auto arg = generate_all_reduce_kernel_argument(1);
        return utils::make_message(
            "    C1 ", arg, "_;\n"
            "    static const int ndim = 1;\n"
            "    typedef Type T;\n"
            "    XINLINE Shape<ndim> shape() const {\n"
            "        return Shape<ndim>(1);\n"
            "    }\n"
            "    XINLINE ", name, ndim, "(\n",
            generate_reduce_kernel_constructor_arguments(1), "\n"
            "        : ", arg, "_(", arg, ") {}\n"
        );
    }

    std::string generate_axis_reduce_kernel_constructor(std::string name, int ndim, int result_ndim) {
        auto arg = generate_all_reduce_kernel_argument(1);
        return utils::make_message(
            "    C1 ", arg, "_;\n"
            "    static const int ndim = ", result_ndim, ";\n"
            "    typedef Type T;\n"
            "    XINLINE Shape<ndim> shape() const {\n"
            "        return ", arg, "_.shape().template axis_reduced_shape<0, ndim>();\n"
            "    }\n"
            "    XINLINE ", name, ndim, "(\n",
            generate_reduce_kernel_constructor_arguments(1), "\n"
            "        : ", arg, "_(", arg, ") {}\n"
        );
    }

    std::string generate_reduce_kernel_caller_code(std::string clsname, std::string funcname, int ndim, int nargs) {
        return utils::make_message(
            generate_reduce_kernel_template_code(nargs),
            "XINLINE ", generate_reduce_kernel_class_signature(clsname, ndim, nargs),
            " ", funcname, "_", ndim, "d(\n",
            generate_reduce_kernel_constructor_arguments(nargs), " {\n",
            "    return ", generate_reduce_kernel_class_signature(clsname, ndim, nargs), "(",
            generate_all_reduce_kernel_arguments(nargs), ");\n",
            "}\n"
        );
    }

    std::string generate_all_reduce_loop(int ndim) {
        return utils::make_message(
            "        T res;\n"
            "        Reducer::SetInitValue(res);\n",
            construct_for_loop(
                ndim,
                utils::make_message(
                    "Reducer::Reduce(res, ",
                    generate_all_reduce_kernel_argument(1), "_[", generate_accessor_string(ndim), "]",
                    ");\n"
                ),
                utils::make_message(generate_all_reduce_kernel_argument(1), "_"),
                8
            ),
            "        return res;\n"
        );
    }

    std::string generate_argument_all_reduce_loop(int ndim) {
        return utils::make_message(
            "        Shape<", ndim, "> idx;"
            "        typename C1::T res;\n"
            "        Reducer::SetInitValue(res);\n",
            construct_for_loop(
                ndim,
                utils::make_message(
                    "typename C1::T tmp = res; Reducer::Reduce(res, ",
                    generate_all_reduce_kernel_argument(1),  "_[", generate_accessor_string(ndim), "]",
                    "); if (tmp != res) idx = Shape<", ndim, ">(", generate_accessor_string(ndim), ");\n"
                ),
                utils::make_message(generate_all_reduce_kernel_argument(1), "_"),
                8
            ),
            "        return indices_to_offset(", generate_all_reduce_kernel_argument(1), "_.shape(), idx);\n"
        );
    }

    std::string construct_axis_reduce_for_loop(int ndim) {
        std::string num_cols = utils::make_message(
                generate_all_reduce_kernel_argument(1), "_.shape()[", ndim - 1, "]");
        return utils::make_message(
            "        T res;\n"
            "        Reducer::SetInitValue(res);\n",
            "        int& i1 = query[", ndim - 1, "];\n"
            "        for (i1 = 0; i1 < ", num_cols, "; ++i1) {\n"
            "            Reducer::Reduce(res, ", generate_all_reduce_kernel_argument(1), "_[query]);\n"
            "        }\n"
            "        return res;\n"
        );
    }

    std::string construct_warp_axis_reduce_for_loop(int ndim) {
        std::string num_cols = utils::make_message(
                generate_all_reduce_kernel_argument(1), "_.shape()[", ndim - 1, "]");
        int x_bits = op::jit::thread_bits();
        return utils::make_message(
            "        __shared__ T buffer[", op::jit::nthreads(), "];\n"
            "        query[", ndim - 1, "] = threadIdx.x;\n"
            "        if (threadIdx.x < ", num_cols, ") {\n"
            "            buffer[threadIdx.x] = ", generate_all_reduce_kernel_argument(1), "_[query];\n"
            "        }\n"
            "        for (unsigned x = blockDim.x; x < ", num_cols, "; x += blockDim.x) {\n"
            "            const int col = x + threadIdx.x;\n"
            "            if (col < ", num_cols, ") {\n"
            "                query[", ndim - 1, "] = col;\n"
            "                Reducer::Reduce(buffer[threadIdx.x], ", generate_all_reduce_kernel_argument(1), "_[query]);\n"
            "            }\n"
            "        }\n"
            "        __syncthreads();\n"
            "        // if number of rows is smaller than buffer,\n"
            "        // fill buffer with neutral value\n"
            "        if (threadIdx.x >= ", num_cols, ") {\n"
            "            Reducer::SetInitValue(buffer[threadIdx.x]);\n"
            "        }\n"
            "        __syncthreads();\n"
            "        ReduceX<Reducer, ", x_bits, ">(buffer, threadIdx.x);\n"
            "        return buffer[0];\n"
        );
    }

    std::string construct_warp_all_reduce_for_loop() {
        return utils::make_message(
            "        __shared__ T buffer[", op::jit::nthreads(), "];\n"
            "        int num_el = ", generate_all_reduce_kernel_argument(1), "_.shape().numel();\n"
            "        if (threadIdx.x < num_el) {\n"
            "            buffer[threadIdx.x] = ", generate_all_reduce_kernel_argument(1), "_[index_to_dim(threadIdx.x, ", generate_all_reduce_kernel_argument(1), "_.shape())];\n"
            "        }\n"
            "        for (unsigned x = blockDim.x; x < num_el; x += blockDim.x) {\n"
            "             const int idx = x + threadIdx.x;\n"
            "             if (idx < num_el) {\n"
            "                Reducer::Reduce(buffer[threadIdx.x], ", generate_all_reduce_kernel_argument(1), "_[index_to_dim(idx, ", generate_all_reduce_kernel_argument(1), "_.shape())]);\n"
            "            }\n"
            "        }\n"
            "        __syncthreads();\n"
            "        // if number of rows is smaller than buffer,\n"
            "        // fill buffer with neutral value\n"
            "        if (threadIdx.x >= num_el) {\n"
            "            Reducer::SetInitValue(buffer[threadIdx.x]);\n"
            "        }\n"
            "        __syncthreads();\n"
            "        ReduceX<Reducer, ", op::jit::thread_bits(), ">(buffer, threadIdx.x);\n"
            "        return buffer[0];\n"
        );
    }

    std::string construct_argument_axis_reduce_for_loop(int ndim) {
        return utils::make_message(
            "        int idx = 0;\n"
            "        typename C1::T res;\n"
            "        Reducer::SetInitValue(res);\n",
            "        int& i1 = query[", ndim - 1, "];\n"
            "        for (i1 = 0; i1 < ", generate_all_reduce_kernel_argument(1), "_.shape()[", ndim - 1, "]; ++i1) {\n"
            "            typename C1::T tmp = res; Reducer::Reduce(res, ",
                         generate_all_reduce_kernel_argument(1), "_[query]",
                         "); if (tmp != res) idx = i1;\n"
            "        }\n"
            "        return idx;\n"
        );
    }

    std::string warp_axis_reduce_prefix_code() {
        return utils::make_message(
            "template<typename Reducer, int x_bits, typename DType>\n"
            "inline __device__ void ReduceX(volatile DType buf[], int tid) {\n"
            "  if (x_bits >= 10) {\n"
            "    if (tid < 512) Reducer::Reduce(buf[tid] , buf[tid + 512]);\n"
            "    __syncthreads();\n"
            "  }\n"
            "  if (x_bits >= 9) {\n"
            "    if (tid < 256) Reducer::Reduce(buf[tid] , buf[tid + 256]);\n"
            "    __syncthreads();\n"
            "  }\n"
            "  if (x_bits >= 8) {\n"
            "    if (tid < 128) Reducer::Reduce(buf[tid] , buf[tid + 128]);\n"
            "    __syncthreads();\n"
            "  }\n"
            "  if (x_bits >= 7) {\n"
            "    if (tid < 64) Reducer::Reduce(buf[tid] , buf[tid + 64]);\n"
            "    __syncthreads();\n"
            "  }\n"
            "  if (x_bits >= 6) {\n"
            "    if (tid < 32) Reducer::Reduce(buf[tid] , buf[tid + 32]);\n"
            "    __syncthreads();\n"
            "  }\n"
            "  // in warp optimization\n"
            "  if (x_bits >= 5) {\n"
            "    if (tid < 16) Reducer::Reduce(buf[tid] , buf[tid + 16]);\n"
            "    __syncthreads();\n"
            "  }\n"
            "  if (x_bits >= 4) {\n"
            "    if (tid < 8) Reducer::Reduce(buf[tid] , buf[tid + 8]);\n"
            "    __syncthreads();\n"
            "  }\n"
            "  if (x_bits >= 3) {\n"
            "    if (tid < 4) Reducer::Reduce(buf[tid] , buf[tid + 4]);\n"
            "    __syncthreads();\n"
            "  }\n"
            "  if (x_bits >= 2) {\n"
            "    if (tid < 2) Reducer::Reduce(buf[tid] , buf[tid + 2]);\n"
            "    __syncthreads();\n"
            "  }\n"
            "  if (x_bits >= 1) {\n"
            "    if (tid < 1) Reducer::Reduce(buf[tid] , buf[tid + 1]);\n"
            "    __syncthreads();\n"
            "  }\n"
            "}\n");
    }
}

typedef std::function<void(const std::string&)> inserter;

void create_all_reduce_kernel_caller(int ndim, inserter insert) {
    insert(utils::make_message(
        generate_reduce_kernel_template_code(1),
        "struct AllReduceKernel", ndim, " {\n",
        generate_all_reduce_kernel_constructor("AllReduceKernel", ndim),
        "    XINLINE T operator[](const Shape<1>&) const {\n",
        generate_all_reduce_loop(ndim),
        "    }\n"
        "};\n"));
    insert(generate_reduce_kernel_caller_code("AllReduceKernel", "all_reduce_kernel", ndim, 1));
}

void create_argument_all_reduce_kernel_caller(int ndim, inserter insert) {
    insert(utils::make_message(
        generate_reduce_kernel_template_code(1),
        "struct ArgumentAllReduceKernel", ndim, " {\n",
        generate_all_reduce_kernel_constructor("ArgumentAllReduceKernel", ndim),
        "    XINLINE T operator[](const Shape<1>&) const {\n",
        generate_argument_all_reduce_loop(ndim),
        "    }\n"
        "};\n"));
    insert(generate_reduce_kernel_caller_code("ArgumentAllReduceKernel", "argument_all_reduce_kernel", ndim, 1));
}

void create_warp_axis_reduce_kernel_caller(int ndim, inserter insert) {
    insert(warp_axis_reduce_prefix_code());
    insert(utils::make_message(
        generate_reduce_kernel_template_code(1),
        "struct WarpAxisReduceKernel", ndim, " {\n",
        generate_axis_reduce_kernel_constructor("WarpAxisReduceKernel", ndim, ndim - 1),
        "    inline __device__ T operator[](const Shape<", ndim - 1, ">& input_query) const {\n"
        "        Shape<", ndim, "> query = input_query.expand_dims(", ndim, ");\n",
        construct_warp_axis_reduce_for_loop(ndim),
        "    }\n"
        "};\n"));
    insert(generate_reduce_kernel_caller_code("WarpAxisReduceKernel", "warp_axis_reduce_kernel", ndim, 1));
}

void create_warp_all_reduce_kernel_caller(int ndim, inserter insert) {
    insert(warp_axis_reduce_prefix_code());
    insert(utils::make_message(
        generate_reduce_kernel_template_code(1),
        "struct WarpAllReduceKernel", ndim, " {\n",
        generate_all_reduce_kernel_constructor("WarpAllReduceKernel", ndim),
        "    inline __device__ T operator[](const Shape<1>&) const {\n",
        construct_warp_all_reduce_for_loop(),
        "    }\n"
        "};\n"));
    insert(generate_reduce_kernel_caller_code("WarpAllReduceKernel", "warp_all_reduce_kernel", ndim, 1));
}

void create_shfl_down_warp_axis_sum_kernel_caller(int ndim, inserter insert) {
    insert("#include <cooperative_groups.h>\n");
    insert(utils::make_message(
        generate_reduce_kernel_template_code(1),
        "struct ShflDownAxisSumKernel", ndim, " {\n",
        generate_axis_reduce_kernel_constructor("ShflDownAxisSumKernel", ndim, ndim - 1),
        "    inline __device__ T operator[](const Shape<", ndim - 1, ">& input_query) const {\n"
        "        Shape<", ndim, "> query = input_query.expand_dims(", ndim, ");\n",
        construct_warp_axis_reduce_for_loop(ndim),
        "    }\n"
        "};\n"));
    insert(generate_reduce_kernel_caller_code("ShflDownAxisSumKernel", "shfl_down_warp_axis_sum", ndim, 1));
}

void create_axis_reduce_kernel_caller(int ndim, inserter insert) {
    insert(utils::make_message(
        generate_reduce_kernel_template_code(1),
        "struct AxisReduceKernel", ndim, " {\n",
        generate_axis_reduce_kernel_constructor("AxisReduceKernel", ndim, ndim - 1),
        "    XINLINE T operator[](const Shape<", ndim - 1, ">& input_query) const {\n"
        "        Shape<", ndim, "> query = input_query.expand_dims(", ndim, ");\n",
        construct_axis_reduce_for_loop(ndim),
        "    }\n"
        "};\n"));
    insert(generate_reduce_kernel_caller_code("AxisReduceKernel", "axis_reduce_kernel", ndim, 1));
}

void create_argument_axis_reduce_kernel_caller(int ndim, inserter insert) {
    insert(utils::make_message(
        generate_reduce_kernel_template_code(1),
        "struct ArgumentAxisReduceKernel", ndim, " {\n",
        generate_axis_reduce_kernel_constructor("ArgumentAxisReduceKernel", ndim, ndim - 1),
        "    XINLINE T operator[](const Shape<", ndim - 1, ">& input_query) const {\n"
        "        Shape<", ndim, "> query = input_query.expand_dims(", ndim, ");\n",
        construct_argument_axis_reduce_for_loop(ndim),
        "    }\n"
        "};\n"));
    insert(generate_reduce_kernel_caller_code("ArgumentAxisReduceKernel", "argument_axis_reduce_kernel", ndim, 1));
}
