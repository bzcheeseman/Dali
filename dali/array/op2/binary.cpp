#include "binary.h"
#include <unordered_map>
#include <string>

#include "dali/utils/tuple_hash.h"
#include "dali/array/array.h"
#include "dali/array/dtype.h"
#include "dali/array/function2/compiler.h"

class Binary {
    public:
        static std::string get_code_template(
                DType dtype,
                OPERATOR_T operator_t,
                const memory::Device& device,
                bool a_contiguous,
                bool b_contiguous,
                bool dst_contiguous,
                int rank) {
            auto cpp_type = dtype_to_cpp_name(dtype);
            std::string code = (utils::MS()
                << "void run(Array dst, Array a, Array b) {\n"
                << "    auto a_view = make" << (a_contiguous ? "_" : "_strided_") << "view<" << cpp_type << ", " << std::to_string(rank) << ">(a);\n"
                << "    auto b_view = make" << (b_contiguous ? "_" : "_strided_") << "view<" << cpp_type << ", " << std::to_string(rank) << ">(b);\n"
                << "    auto dst_view = make" << (dst_contiguous ? "_" : "_strided_") << "view<" << cpp_type + ", " << std::to_string(rank) << ">(dst);\n"
                << "    int num_el = dst.number_of_elements();\n"
            );
            std::string for_loop;
            if (rank == 1) {
                for_loop = "    for (int i = 0; i < num_el; ++i) {\n"
                           "        dst_view(i) " + operator_to_name(operator_t) + " a_view(i) + b_view(i);\n"
                           "    }\n}\n";
            } else {
                for_loop = "    for (int i = 0; i < num_el; ++i) {\n"
                           "        auto query = index_to_dim(i, dst_view.shape());\n"
                           "        dst_view[query] " + operator_to_name(operator_t) + " a_view[query] + b_view[query];\n"
                           "    }\n}\n";
            }
            code += for_loop;
            return code;
        }

        static macro_args_t get_macro_args(
                DType dtype,
                OPERATOR_T operator_t,
                const memory::Device& device,
                bool a_contiguous,
                bool b_contiguous,
                bool dst_contiguous,
                int rank) {
            return {
                { "DALI_ARRAY_HIDE_LAZY", "1"},
            };
        }

        static hash_t get_hash(
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
                    1, // Binary add symbol
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


namespace op2 {
    Assignable<Array> add(Array a, Array b) {
        return Assignable<Array>([a, b](Array& out, const OPERATOR_T& operator_t) {
            ASSERT2(a.dtype() == b.dtype(), "dtypes don't match");
            ASSERT2(a.ndim() == b.ndim(), "ranks don't match");
            std::vector<int> output_bshape = a.bshape();
            auto output_dtype = a.dtype();
            auto b_bshape = b.bshape();
            auto output_device = a.preferred_device();
            for (size_t dim = 0; dim < a.ndim(); dim++) {
                ASSERT2(
                    (output_bshape[dim] == b_bshape[dim]) ||
                    (output_bshape[dim] == -1 || b_bshape[dim] == -1),
                    "shapes dont match"
                );
                if (b_bshape[dim] != -1) {
                    output_bshape[dim] = b_bshape[dim];
                }
            }

            if (out.is_stateless()) {
                for (auto& dim : output_bshape) {
                    if (dim < -1) {
                        dim = std::abs(dim);
                    }
                }

                // HACK: should check shape, type etc.. more thoroughly
                out.initialize_with_bshape(output_bshape,
                                           output_dtype,
                                           output_device);
            } else {
                ASSERT2(output_dtype == out.dtype(), "output type is wrong.");
                auto out_bshape = out.bshape();
                // HACK no broadcasting until strides return.
                for (size_t dim = 0; dim < output_bshape.size(); dim++) {
                    ASSERT2(
                        (std::abs(output_bshape[dim]) == std::abs(out_bshape[dim])),
                        "output shape doesn't match computation shape"
                    );
                }
            }

            bool a_contiguous = a.strides().empty();
            bool b_contiguous = b.strides().empty();
            bool dst_contiguous = out.strides().empty();
            int op_rank = (
                (a_contiguous && b_contiguous && dst_contiguous) ?
                1 : output_bshape.size()
            );

            hash_t hash = Binary::get_hash(
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
                    output_dtype,
                    operator_t,
                    output_device,
                    a_contiguous,
                    b_contiguous,
                    dst_contiguous,
                    op_rank
                );
                auto code_template = Binary::get_code_template(
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
            f_ptr(out, a, b);
        });
    }
}
