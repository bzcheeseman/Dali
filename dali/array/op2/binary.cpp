#include "binary.h"
#include <unordered_map>
#include <string>

#include "dali/utils/tuple_hash.h"
#include "dali/array/array.h"
#include "dali/array/dtype.h"
#include "dali/array/function2/compiler.h"

class Binary {
    public:
        static std::string get_code_template() {
            return "void run(Array dst, Array a, Array b) {\n"
                   "    auto a_view   = make_view<DALI_RTC_TYPE, 1>(a);\n"
                   "    auto b_view   = make_view<DALI_RTC_TYPE, 1>(b);\n"
                   "    auto dst_view = make_view<DALI_RTC_TYPE, 1>(dst);\n"
                   "\n"
                   "    int num_el = dst.number_of_elements();\n"
                   "\n"
                   "    for (int i = 0; i < num_el; ++i) {\n"
                   "        dst_view(i) DALI_RTC_OPERATOR a_view(i) + b_view(i);\n"
                   "    }\n"
                   "}\n";
        }

        static macro_args_t get_macro_args(
                DType dtype,
                OPERATOR_T operator_t,
                const memory::Device& device) {
            return {
                { "DALI_RTC_TYPE", dtype_to_cpp_name(dtype) },
                { "DALI_RTC_OPERATOR", operator_to_name(operator_t) },
                { "DALI_RTC_GPU", device.is_cpu() ? "0" : "1" },
                { "DALI_ARRAY_HIDE_LAZY", "1"}
            };
        }

        static hash_t get_hash(
                DType dtype,
                OPERATOR_T operator_t,
                const memory::Device& device) {
            // TODO(szymon): make more general.
            return utils::get_hash(std::make_tuple(1, dtype, operator_t, device.is_cpu()));
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
            hash_t hash = Binary::get_hash(output_dtype, operator_t, output_device);

            if (!array_op_compiler.load(hash)) {
                auto macro_args = Binary::get_macro_args(
                    output_dtype,
                    operator_t,
                    output_device
                );
                auto code_template = Binary::get_code_template();
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
