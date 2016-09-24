#include "elementwise_kernel_utils.h"
#include "dali/utils/make_message.h"
#include "dali/utils/print_utils.h"
#include "dali/array/op2/rtc_utils.h"

namespace {
    std::string generate_all_reduce_kernel_class_signature(std::string name, int ndim) {
        return utils::make_message(name, ndim, "<Reducer, Type, C1>");
    }

    std::string generate_all_reduce_kernel_argument() {
        return "arg_1_view";
    }

    std::string generate_all_reduce_kernel_template_code() {
        return "template<typename Reducer, typename Type, typename C1>\n";
    }

    std::string generate_all_reduce_kernel_constructor_arguments() {
        return utils::make_message("        const C1& ", generate_all_reduce_kernel_argument(), ")");
    }

    std::string generate_all_reduce_kernel_constructor(std::string name, int ndim, int result_ndim) {
        auto arg = generate_all_reduce_kernel_argument();
        return utils::make_message(
            "    const C1& ", arg, "_;\n"
            "    static const int ndim = ", result_ndim, ";\n"
            "    typedef Type T;\n"
            "    XINLINE const Shape<ndim>& shape() const {\n"
            "        return ", generate_all_reduce_kernel_argument(), "_.shape();\n"
            "    }\n"
            "    XINLINE ", name, ndim, "(\n",
            generate_all_reduce_kernel_constructor_arguments(), "\n"
            "        : ", arg, "_(", arg, ") {}\n"
        );
    }

    std::string generate_all_reduce_kernel_caller_code(std::string clsname, std::string funcname, int ndim) {
        return utils::make_message(
            generate_all_reduce_kernel_template_code(),
            "XINLINE ", generate_all_reduce_kernel_class_signature(clsname, ndim),
            " ", funcname, "_", ndim, "d(\n",
            generate_all_reduce_kernel_constructor_arguments(), " {\n",
            "    return ", generate_all_reduce_kernel_class_signature(clsname, ndim), "(",
            generate_all_reduce_kernel_argument(), ");\n",
            "}\n"
        );
    }
}

std::string generate_all_reduce_loop(int ndim) {
    return utils::make_message(
        "        T res;\n"
        "        Reducer::SetInitValue(res);\n",
        construct_for_loop(
            ndim,
            utils::make_message(
                "Reducer::Reduce(res, ",
                generate_all_reduce_kernel_argument(), "_[query]",
                ");\n"
            ),
            utils::make_message(generate_all_reduce_kernel_argument(), "_"),
            8
        ),
        "        return res;\n"
     );
 }

std::string generate_argument_all_reduce_loop(int ndim) {
    return utils::make_message(
        "        Shape<ndim> idx;"
        "        typename C1::T res;\n"
        "        Reducer::SetInitValue(res);\n",
        construct_for_loop(
            ndim,
            utils::make_message(
                "typename C1::T tmp = res; Reducer::Reduce(res, ",
                generate_all_reduce_kernel_argument(), "_[query]",
                "); if (tmp != res) idx = query;\n"
            ),
            utils::make_message(generate_all_reduce_kernel_argument(), "_"),
            8
        ),
        "        return indices_to_offset(shape(), idx);\n"
    );
}

std::string create_all_reduce_kernel_caller(int ndim, int result_ndim) {
    return utils::make_message(
        generate_all_reduce_kernel_template_code(),
        "struct AllReduceKernel", ndim, " {\n",
        generate_all_reduce_kernel_constructor("AllReduceKernel", ndim, result_ndim),
        "    XINLINE T operator[](Shape<", result_ndim, ">) const {\n",
        generate_all_reduce_loop(ndim),
        "    }\n"
        "    XINLINE T operator()(int) const {\n",
        generate_all_reduce_loop(ndim),
        "    }\n"
        "};\n",
        generate_all_reduce_kernel_caller_code("AllReduceKernel", "all_reduce_kernel", ndim)
    );
}

std::string create_argument_all_reduce_kernel_caller(int ndim, int result_ndim) {
    return utils::make_message(
        generate_all_reduce_kernel_template_code(),
        "struct ArgumentAllReduceKernel", ndim, " {\n",
        generate_all_reduce_kernel_constructor("ArgumentAllReduceKernel", ndim, result_ndim),
        "    XINLINE T operator[](Shape<", result_ndim, ">) const {\n",
        generate_argument_all_reduce_loop(ndim),
        "    }\n"
        "    XINLINE T operator()(int) const {\n",
        generate_argument_all_reduce_loop(ndim),
        "    }\n"
        "};\n",
        generate_all_reduce_kernel_caller_code("ArgumentAllReduceKernel", "argument_all_reduce_kernel", ndim)
    );
}
