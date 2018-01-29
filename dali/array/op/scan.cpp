#include "scan.h"

#include "dali/array/jit/jit_runner.h"
#include "dali/array/jit/scalar_view.h"
#include "dali/utils/make_message.h"

namespace {
    std::vector<int> pad_last(std::vector<int> shape) {
        if (shape.size() == 0) {
            return {2};
        }
        shape.back() += 1;
        return shape;
    }
}

namespace op {
    namespace jit {
        struct AxisScan : public JITNode {
            std::string functor_name_;
            bool inclusive_;
            AxisScan(std::string functor_name, Array array, DType dtype, bool inclusive) :
                     JITNode(inclusive ? array.shape() : pad_last(array.shape()),
                             dtype, {array,}), functor_name_(functor_name), inclusive_(inclusive) {
            }

            std::string base_kernel_name() const {
                return utils::make_message("axis_scan_", inclusive_ ? "inclusive" : "exclusive");
            }

            std::string kernel_name() const {
                return utils::make_message(
                  base_kernel_name(), "<", functor_name_, ", " , dtype_to_cpp_name(dtype_), ">");
            }

            virtual bool chainable() const override {
                return false;
            }

            virtual bool grid_keep_inner_dim() const override {
                return false;
            }

            virtual PARALLELISM_T parallelism_type() const override {
                return INDEPENDENT_BLOCK;
            }

            void compilation_parameters(utils::Hasher& hasher) const override {
                hasher.add(functor_name_).add(inclusive_);
            }

            std::string prefix_code(memory::DeviceT device_type) const override {
                int rank = std::max(1, ndim());
                std::string clsname = utils::make_message(char(std::toupper(base_kernel_name()[0])),
                                      base_kernel_name().substr(1));
                std::string exclusive_set_init_element = "";
                if (!inclusive_) {
                    exclusive_set_init_element = (
                        "                query[ndim - 1] = 0;\n"
                        "                output_[query] = init;\n");
                }
                return utils::make_message(
                    "template<typename Reducer, typename Type, typename C1, typename C2>\n"
                    "struct ", clsname, " {\n"
                    "    C1 arg_;\n"
                    "    C2 output_;\n"
                    "    static const int ndim = C1::ndim;\n"
                    "    typedef Type T;\n"
                    "    XINLINE Shape<ndim> shape() const {\n"
                    "        return output_.shape();\n"
                    "    }\n"
                    "    XINLINE ", clsname, "(C1 arg, C2 output) : arg_(arg), output_(output) {}\n"
                    "    inline __device__ void operator[](Shape<ndim> query) {\n"
                    "        T init;\n"
                    "        int row_size = arg_.shape()[ndim - 1];\n"
                    "        Reducer::SetInitValue(init);\n"
                    "        __shared__ T buffer[", 2 * op::jit::nthreads(), "];\n"
                    "        T block_total = init;\n"
                    "        // Perform scan on one block at a time, keeping track of the total value of\n"
                    "        // all blocks processed so far.\n"
                    "        for (unsigned block_col = 0; block_col < row_size; block_col += ", 2 * op::jit::nthreads(), ") {\n"
                    "            // Load data into shared memory (two values per thread).\n"
                    "            query[ndim - 1] = block_col + threadIdx.x;\n"
                    "            if (query[ndim - 1] < row_size) {\n"
                    "                buffer[threadIdx.x] = arg_[query];\n"
                    "            } else {\n"
                    "                buffer[threadIdx.x] = init;\n"
                    "            }\n"
                    "            query[ndim - 1] = block_col + ", op::jit::nthreads(), " + threadIdx.x;\n"
                    "            if (query[ndim - 1] < row_size) {\n"
                    "                buffer[", op::jit::nthreads(), " + threadIdx.x] = arg_[query];\n"
                    "            } else {\n"
                    "                buffer[", op::jit::nthreads(), " + threadIdx.x] = init;\n"
                    "            }\n"
                    "            // Add the total value of all previous blocks to the first value of this block.\n"
                    "            if (threadIdx.x == 0) {\n", exclusive_set_init_element,
                    "                Reducer::Reduce(buffer[0], block_total);\n"
                    "            }\n"
                    "            __syncthreads();\n"
                    "            // Parallel reduction (up-sweep).\n"
                    "            for (unsigned s = ", op::jit::nthreads(), ", d = 1; s >= 1; s >>= 1, d <<= 1) {\n"
                    "                if (threadIdx.x < s) {\n"
                    "                    unsigned offset = (2 * threadIdx.x + 1) * d - 1;\n"
                    "                    Reducer::Reduce(buffer[offset + d], buffer[offset]);\n"
                    "                }\n"
                    "                __syncthreads();\n"
                    "            }\n"
                    "            // Down-sweep.\n"
                    "            for (unsigned s = 2, d = ", op::jit::nthreads() / 2, "; d >= 1; s <<= 1, d >>= 1) {\n"
                    "                if (threadIdx.x < s - 1) {\n"
                    "                    unsigned offset = 2 * (threadIdx.x + 1) * d - 1;\n"
                    "                    Reducer::Reduce(buffer[offset + d], buffer[offset]);\n"
                    "                }\n"
                    "                __syncthreads();\n"
                    "            }\n"
                    "            // Write back to output.\n"
                    "            if (threadIdx.x < row_size) {\n"
                    "                query[ndim - 1] = ", (inclusive_ ? "" : "1 + "), "threadIdx.x;\n"
                    "                output_[query] = buffer[threadIdx.x];\n"
                    "            }\n"
                    "            if (", op::jit::nthreads(), " + threadIdx.x < row_size) {\n"
                    "                query[ndim - 1] = ", (inclusive_ ? 0 : 1) + op::jit::nthreads(), " + threadIdx.x;\n"
                    "                output_[query] = buffer[", op::jit::nthreads(), " + threadIdx.x];\n"
                    "            }\n"
                    "            block_total = buffer[", 2 * op::jit::nthreads() - 1, "];\n"
                    "            __syncthreads();\n"
                    "        }\n"
                    "    }\n"
                    "};\n"
                    "template<typename Reducer, typename Type, typename C1, typename C2>\n"
                    "XINLINE ", clsname, "<Reducer, Type, C1, C2> ", base_kernel_name(), "(\n"
                    "        C1 arg, C2 output) {\n"
                    "    return ", clsname, "<Reducer, Type, C1, C2>(arg, output);\n"
                    "}\n"
                );
            }

            expression_ptr copy() const override {
                return std::make_shared<AxisScan>(
                    functor_name_, arguments_[0], dtype_, inclusive_);
            }

            std::string get_call_code_nd(const SymbolTable& symbol_table,
                                         memory::DeviceT device_type) const override {
                return utils::make_message(kernel_name(), "(",
                    op::jit::get_call_code_nd(arguments_[0], symbol_table, device_type), ", ",
                    symbol_table.get_temporary_name(this), ")");
            }
        };
    }
    Array axis_scan(const Array& a,
                    const std::string& reducer_name,
                    bool inclusive) {
        return Array(std::make_shared<op::jit::AxisScan>(
                reducer_name, a, a.dtype(), inclusive));
    }
    Array cumsum(const Array& a, bool inclusive) {
        return axis_scan(a, "reducers::sum", inclusive);
    }
    Array cumprod(const Array& a, bool inclusive) {
        return axis_scan(a, "reducers::product", inclusive);
    }
}
