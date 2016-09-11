#include "binary.h"
#include <unordered_map>
#include <string>

#include "dali/utils/tuple_hash.h"
#include "dali/array/array.h"
#include "dali/array/dtype.h"
#include "dali/array/function2/compiler.h"

std::string build_view_constructor(
        const std::string& cpp_type,
        bool a_contiguous,
        bool b_contiguous,
        bool dst_contiguous,
        int rank) {
    return utils::MS()
        << "    auto a_view = make" << (a_contiguous ? "_" : "_strided_")
        << "view<" << cpp_type << ", " << rank << ">(a);\n"
        << "    auto b_view = make" << (b_contiguous ? "_" : "_strided_")
        << "view<" << cpp_type << ", " << rank << ">(b);\n"
        << "    auto dst_view = make" << (dst_contiguous ? "_" : "_strided_")
        << "view<" << cpp_type + ", " << rank << ">(dst);\n";
}

class Binary {
    public:
        static std::string get_code_template(
                const std::string& fname,
                DType dtype,
                OPERATOR_T operator_t,
                const memory::Device& device,
                bool a_contiguous,
                bool b_contiguous,
                bool dst_contiguous,
                int rank) {
            auto cpp_type = dtype_to_cpp_name(dtype);
            std::string code = "void run(Array dst, Array a, Array b) {\n";
            code += build_view_constructor(
                cpp_type, a_contiguous, b_contiguous, dst_contiguous, rank
            );
            code += "    int num_el = dst.number_of_elements();\n";
            std::string for_loop;
            if (rank == 1) {
                for_loop = "    for (int i = 0; i < num_el; ++i) {\n"
                           "        dst_view(i) " + operator_to_name(operator_t) + " " + fname + "<" + cpp_type + ">::Map(a_view(i), b_view(i));\n"
                           "    }\n}\n";
            } else {
                // TODO: make the for loop increase in nesting to avoid usage of
                // index_to_dim (because division == expensive)
                for_loop = "    for (int i = 0; i < num_el; ++i) {\n"
                           "        auto query = index_to_dim(i, dst_view.shape());\n"
                           "        dst_view[query] " + operator_to_name(operator_t) + " " + fname + "<" + cpp_type + ">::Map(a_view[query], b_view[query]);\n"
                           "    }\n}\n";
            }
            code += for_loop;
            return code;
        }

        static macro_args_t get_macro_args(
                const std::string& fname,
                DType dtype,
                OPERATOR_T operator_t,
                const memory::Device& device,
                bool a_contiguous,
                bool b_contiguous,
                bool dst_contiguous,
                int rank) {
            return {};
        }

        static hash_t get_hash(
                const std::string& fname,
                DType dtype,
                OPERATOR_T operator_t,
                const memory::Device& device,
                bool a_contiguous,
                bool b_contiguous,
                bool dst_contiguous,
                int rank) {
            // TODO(szymon): make more general.
            return utils::get_hash(
                std::make_tuple(
                    fname,
                    dtype,
                    operator_t,
                    device.is_cpu(),
                    a_contiguous,
                    b_contiguous,
                    dst_contiguous,
                    rank
                )
            );
        }
};

class BinaryKernel {
    public:
        static std::string get_code_template(
                const std::string& kernel_name,
                const std::string& kernel,
                DType dtype,
                OPERATOR_T operator_t,
                const memory::Device& device,
                bool a_contiguous,
                bool b_contiguous,
                bool dst_contiguous,
                int rank) {
            auto cpp_type = dtype_to_cpp_name(dtype);
            std::string code = kernel + "void run(Array dst, Array a, Array b) {\n";
            code += build_view_constructor(
                cpp_type, a_contiguous, b_contiguous, dst_contiguous, rank
            );
            code += "    int num_el = dst.number_of_elements();\n";
            std::string for_loop;
            if (rank == 1) {
                for_loop = "    for (int i = 0; i < num_el; ++i) {\n"
                           "        dst_view(i) " + operator_to_name(operator_t) + " " + kernel_name + "(a_view, b_view, i);\n"
                           "    }\n}\n";
            } else {
                // TODO: make the for loop increase in nesting to avoid usage of
                // index_to_dim (because division == expensive)
                for_loop = "    for (int i = 0; i < num_el; ++i) {\n"
                           "        auto query = index_to_dim(i, dst_view.shape());\n"
                           "        dst_view[query] " + operator_to_name(operator_t) + " " + kernel_name + "(a_view, b_view, query);\n"
                           "    }\n}\n";
            }
            code += for_loop;
            return code;
        }

        static macro_args_t get_macro_args(
                const std::string& kernel_name,
                const std::string& kernel,
                DType dtype,
                OPERATOR_T operator_t,
                const memory::Device& device,
                bool a_contiguous,
                bool b_contiguous,
                bool dst_contiguous,
                int rank) {
            return {};
        }

        static hash_t get_hash(
                const std::string& kernel_name,
                const std::string& kernel,
                DType dtype,
                OPERATOR_T operator_t,
                const memory::Device& device,
                bool a_contiguous,
                bool b_contiguous,
                bool dst_contiguous,
                int rank) {
            // TODO(szymon): make more general.
            return utils::get_hash(
                std::make_tuple(
                    kernel_name,
                    dtype,
                    operator_t,
                    device.is_cpu(),
                    a_contiguous,
                    b_contiguous,
                    dst_contiguous,
                    rank
                )
            );
        }
};

namespace {
    void initialize_output_array(Array& out,
                                 const DType& output_dtype,
                                 const memory::Device& output_device,
                                 std::vector<int>* output_bshape_ptr) {
        auto& output_bshape = *output_bshape_ptr;
        if (out.is_stateless()) {
            // when constructing a stateless
            // output, we decide what the output
            // shape will be. Broadcasted greater
            // than one dimensions are expanded
            // out:
            for (auto& dim : output_bshape) {
                if (dim < -1) {
                    dim = std::abs(dim);
                }
            }

            out.initialize_with_bshape(output_bshape,
                                       output_dtype,
                                       output_device);
        } else {
            bool broadcast_reshaped_output = false;

            for (const int& dim_size: out.bshape()) {
                if (dim_size < -1) {
                    broadcast_reshaped_output = true;
                    break;
                }
            }

            ASSERT2(!broadcast_reshaped_output,
                    "Cannot assign to broadcasted output with broadcasted dimension"
                    " bigger than 1, because it results in many-to-one mappings.");


            bool output_bshape_compatible = out.ndim() == output_bshape.size();
            if (output_bshape_compatible) {
                for (int i = 0; i < out.ndim(); ++i) {
                    if (output_bshape[i] != -1 && std::abs(output_bshape[i]) != out.shape()[i]) {
                        output_bshape_compatible = false;
                        break;
                    }
                }
            }

            ASSERT2(output_bshape_compatible,
                    utils::MS() << "Cannot assign result of shape " << output_bshape
                                << " to a location of shape " << out.shape() << ".");
            ASSERT2(out.dtype() == output_dtype,
                    utils::MS() << "Cannot assign result of dtype " << output_dtype
                                << " to a location of dtype " << out.dtype() << ".");
        }
    }

    std::vector<int> get_binary_function_bshape(
            const std::vector<int>& left_shape,
            const std::vector<int>& right_shape) {
        std::vector<int> output_bshape = left_shape;
        for (size_t dim = 0; dim < output_bshape.size(); dim++) {
            ASSERT2(
                (output_bshape[dim] == right_shape[dim]) ||
                (output_bshape[dim] == -1 || right_shape[dim] == -1),
                "shapes dont match"
            );
            if (right_shape[dim] != -1) {
                output_bshape[dim] = right_shape[dim];
            }
        }
        return output_bshape;
    }

    Assignable<Array> elementwise(
            const Array& a,
            const Array& b,
            const char* functor_name) {
        ASSERT2(a.dtype() == b.dtype(), "dtypes don't match");
        ASSERT2(a.ndim() == b.ndim(), "ranks don't match");
        auto output_bshape = get_binary_function_bshape(a.bshape(), b.bshape());
        return Assignable<Array>([a, b, output_bshape, functor_name](
                Array& out,
                const OPERATOR_T& operator_t) mutable {
            auto output_dtype = a.dtype();
            // check what good device to use given a,b
            auto output_device = ReduceOverArgs<DeviceReducer>::reduce(a, b);
            initialize_output_array(
                out,
                output_dtype,
                output_device,
                &output_bshape
            );
            // once out has been checked, we can now include it in the
            // device deduction process
            output_device = ReduceOverArgs<DeviceReducer>::reduce(out, a, b);

            // RTC specific logic:
            bool a_contiguous = a.strides().empty();
            bool b_contiguous = b.strides().empty();
            bool dst_contiguous = out.strides().empty();
            int op_rank = (
                (a_contiguous && b_contiguous && dst_contiguous) ?
                1 : output_bshape.size()
            );
            hash_t hash = Binary::get_hash(
                functor_name,
                output_dtype,
                operator_t,
                output_device,
                a_contiguous,
                b_contiguous,
                dst_contiguous,
                op_rank
            );

            if (!array_op_compiler.load(hash)) {
                auto macro_args = Binary::get_macro_args(
                    functor_name,
                    output_dtype,
                    operator_t,
                    output_device,
                    a_contiguous,
                    b_contiguous,
                    dst_contiguous,
                    op_rank
                );
                auto code_template = Binary::get_code_template(
                    functor_name,
                    output_dtype,
                    operator_t,
                    output_device,
                    a_contiguous,
                    b_contiguous,
                    dst_contiguous,
                    op_rank
                );
                array_op_compiler.compile<Array,Array,Array>(
                    hash,
                    code_template,
                    macro_args
                );
            }
            auto f_ptr = array_op_compiler.get_function<Array,Array,Array>(hash);
            if (op_rank == 1 && output_bshape.size() != 1) {
                f_ptr(out.copyless_ravel(), a.copyless_ravel(), b.copyless_ravel());
            } else {
                f_ptr(out, a, b);
            }
        });
    }

    Assignable<Array> binary_kernel_function(
            const Array& a,
            const Array& b,
            const char* function_name,
            const char* kernel) {
        ASSERT2(a.dtype() == b.dtype(), "dtypes don't match");
        ASSERT2(a.ndim() == b.ndim(), "ranks don't match");
        auto output_bshape = get_binary_function_bshape(a.bshape(), b.bshape());
        return Assignable<Array>([a, b, output_bshape, function_name, kernel](
                Array& out,
                const OPERATOR_T& operator_t) mutable {
            auto output_dtype = a.dtype();
            // check what good device to use given a,b
            auto output_device = ReduceOverArgs<DeviceReducer>::reduce(a, b);
            initialize_output_array(
                out,
                output_dtype,
                output_device,
                &output_bshape
            );
            // once out has been checked, we can now include it in the
            // device deduction process
            output_device = ReduceOverArgs<DeviceReducer>::reduce(out, a, b);

            // RTC specific logic:
            bool a_contiguous = a.strides().empty();
            bool b_contiguous = b.strides().empty();
            bool dst_contiguous = out.strides().empty();
            int op_rank = (
                (a_contiguous && b_contiguous && dst_contiguous) ?
                2 : output_bshape.size()
            );
            hash_t hash = BinaryKernel::get_hash(
                function_name,
                kernel,
                output_dtype,
                operator_t,
                output_device,
                a_contiguous,
                b_contiguous,
                dst_contiguous,
                op_rank
            );

            if (!array_op_compiler.load(hash)) {
                auto macro_args = BinaryKernel::get_macro_args(
                    function_name,
                    kernel,
                    output_dtype,
                    operator_t,
                    output_device,
                    a_contiguous,
                    b_contiguous,
                    dst_contiguous,
                    op_rank
                );
                auto code_template = BinaryKernel::get_code_template(
                    function_name,
                    kernel,
                    output_dtype,
                    operator_t,
                    output_device,
                    a_contiguous,
                    b_contiguous,
                    dst_contiguous,
                    op_rank
                );
                array_op_compiler.compile<Array,Array,Array>(
                    hash,
                    code_template,
                    macro_args
                );
            }

            auto a_reshaped = a.reshape_broadcasted(output_bshape);
            auto b_reshaped = b.reshape_broadcasted(output_bshape);

            auto f_ptr = array_op_compiler.get_function<Array,Array,Array>(hash);
            if (op_rank == 2 && output_bshape.size() != 2) {
                int low_dim = output_bshape.size() > 0 ? output_bshape.back() : 1;
                f_ptr(
                    out.copyless_reshape({-1, low_dim}),
                    a_reshaped.copyless_reshape({-1, low_dim}),
                    b_reshaped.copyless_reshape({-1, low_dim})
                );
            } else {
                f_ptr(out, a_reshaped, b_reshaped);
            }
        });
    }
}

namespace op2 {
    Assignable<Array> add(const Array& a, const Array& b) {
        return elementwise(a, b, "functor::add");
    }

    Assignable<Array> sub(const Array& a, const Array& b) {
        return elementwise(a, b, "functor::sub");
    }

    Assignable<Array> eltmul(const Array& a, const Array& b) {
        return elementwise(a, b, "functor::eltmul");
    }

    Assignable<Array> eltdiv(const Array& a, const Array& b) {
        return elementwise(a, b, "functor::eltdiv");
    }

    Assignable<Array> pow(const Array& a, const Array& b) {
        return elementwise(a, b, "functor::power");
    }

    Assignable<Array> equals(const Array& a, const Array& b) {
        return elementwise(a, b, "functor::equals");
    }

    Assignable<Array> prelu(const Array& x, const Array& weights) {
        return elementwise(x, weights, "functor::prelu");
    }

    Assignable<Array> circular_convolution(const Array& x, const Array& weights) {
        return binary_kernel_function(
            x,
            weights,
            "circular_convolution_kernel",
            "template<template <typename,int> class C1,\n"
            "         template <typename,int> class C2,\n"
            "         typename T, int ndim>\n"
            "XINLINE T circular_convolution_kernel(\n"
            "         const C1<T, ndim> a_view,\n"
            "         const C2<T, ndim> b_view,\n"
            "         Shape<ndim> query) {\n"
            "    T res = static_cast<T>(0);\n"
            "    const int conv_size = b_view.shape()[ndim - 1];\n"
            "    const int x = query[ndim - 1];\n"
            "    Shape<ndim> a_query = query;\n"
            "    Shape<ndim> b_query = query;\n"
            "    int& shift_idx = b_query[ndim - 1];\n"
            "    int& offset = a_query[ndim - 1];\n"
            "    for (shift_idx = 0; shift_idx < conv_size; shift_idx++) {\n"
            "        offset = x + shift_idx;\n"
            "        if (offset >= conv_size) {\n"
            "            offset -= conv_size;\n"
            "        }\n"
            "        res += a_view[a_query] * b_view[b_query];\n"
            "    }\n"
            "    return res;\n"
            "}\n"
        );
    }
}
