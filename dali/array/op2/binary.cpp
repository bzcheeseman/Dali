
#include <unordered_map>
#include <string>

#include "dali/array/array.h"
#include "dali/array/dtype.h"
#include "dali/array/function2/compiler.h"

class Binary {
  public:
    static std::string get_code_template() {
        return "void run(Array dst, Array a, Array b) {"
               "    auto a_view   = make_view<DALI_RTC_TYPE, 1>(a);"
               "    auto b_view   = make_view<DALI_RTC_TYPE, 1>(b);"
               "    auto dst_view = make_view<DALI_RTC_TYPE, 1>(dst);"
               ""
               "    int num_el = dst.number_of_elements();"
               ""
               "    for (int i = 0; i < num_el; ++i) {"
               "        dst_view(i) = a_view(i) + b_view(i);"
               "    }"
               "}";
    }

    static macro_args_t get_macro_args(DType dtype) {
        return {
            { "TYPE", dtype_to_cpp_name(dtype) }
        };
    }

    static hash_t get_hash(DType dtype) {
        // TODO(szymon): todo.
        return 1;
    }
};


namespace op2 {
    void add(Array dst, Array a, Array b) {
        auto dtype = dst.dtype();

        hash_t hash = Binary::get_hash(dtype);

        if (!chief_compiler.load(hash)) {
            auto macro_args    = Binary::get_macro_args(dtype);
            auto code_template = Binary::get_code_template();
            chief_compiler.compile<Array,Array,Array>(hash, code_template, macro_args);
        }

        auto f_ptr = chief_compiler.get_function<Array,Array,Array>(hash);

        f_ptr(dst, a, b);
    }

}
