#include "elementwise_kernel_utils.h"
#include "dali/utils/make_message.h"
#include "dali/utils/print_utils.h"
#include "dali/array/op2/rtc_utils.h"

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
            if (i + 1 != nargs) {
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
            stream << "        const C" << (i + 1) << "& "
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

    std::string generate_all_reduce_kernel_constructor(std::string name, int ndim, int result_ndim) {
        auto arg = generate_all_reduce_kernel_argument(1);
        return utils::make_message(
            "    const C1& ", arg, "_;\n"
            "    static const int ndim = ", result_ndim, ";\n"
            "    typedef Type T;\n"
            "    XINLINE const Shape<ndim>& shape() const {\n"
            "        return ", arg, "_.shape();\n"
            "    }\n"
            "    XINLINE ", name, ndim, "(\n",
            generate_reduce_kernel_constructor_arguments(1), "\n"
            "        : ", arg, "_(", arg, ") {}\n"
        );
    }

    std::string generate_axis_reduce_kernel_constructor(std::string name, int ndim, int result_ndim) {
        auto arg = generate_all_reduce_kernel_argument(1);
        return utils::make_message(
            "    const C1& ", arg, "_;\n"
            "    static const int ndim = ", result_ndim, ";\n"
            "    typedef Type T;\n"
            "    XINLINE Shape<ndim> shape() const {\n"
            "        return ", arg, "_.shape().axis_reduced_shape();\n"
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
                    generate_all_reduce_kernel_argument(1), "_[query]",
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
            "        Shape<ndim> idx;"
            "        typename C1::T res;\n"
            "        Reducer::SetInitValue(res);\n",
            construct_for_loop(
                ndim,
                utils::make_message(
                    "typename C1::T tmp = res; Reducer::Reduce(res, ",
                    generate_all_reduce_kernel_argument(1), "_[query]",
                    "); if (tmp != res) idx = query;\n"
                ),
                utils::make_message(generate_all_reduce_kernel_argument(1), "_"),
                8
            ),
            "        return indices_to_offset(shape(), idx);\n"
        );
    }

    std::string construct_axis_reduce_for_loop(int ndim) {
        return utils::make_message(
            "        T res;\n"
            "        Reducer::SetInitValue(res);\n",
            "        int& i1 = query[", ndim - 1, "];\n"
            "        for (i1 = 0; i1 < ", generate_all_reduce_kernel_argument(1), "_.shape()[", ndim - 1, "]; ++i1) {\n"
            "            Reducer::Reduce(res, ", generate_all_reduce_kernel_argument(1), "_[query]);\n"
            "        }\n"
            "        return res;\n"
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
}


std::string create_all_reduce_kernel_caller(int ndim, int result_ndim) {
    return utils::make_message(
        generate_reduce_kernel_template_code(1),
        "struct AllReduceKernel", ndim, " {\n",
        generate_all_reduce_kernel_constructor("AllReduceKernel", ndim, result_ndim),
        "    XINLINE T operator[](Shape<", result_ndim, ">) const {\n",
        generate_all_reduce_loop(ndim),
        "    }\n"
        "    XINLINE T operator()(int) const {\n",
        generate_all_reduce_loop(ndim),
        "    }\n"
        "};\n",
        generate_reduce_kernel_caller_code("AllReduceKernel", "all_reduce_kernel", ndim, 1)
    );
}

std::string create_argument_all_reduce_kernel_caller(int ndim, int result_ndim) {
    return utils::make_message(
        generate_reduce_kernel_template_code(1),
        "struct ArgumentAllReduceKernel", ndim, " {\n",
        generate_all_reduce_kernel_constructor("ArgumentAllReduceKernel", ndim, result_ndim),
        "    XINLINE T operator[](Shape<", result_ndim, ">) const {\n",
        generate_argument_all_reduce_loop(ndim),
        "    }\n"
        "    XINLINE T operator()(int) const {\n",
        generate_argument_all_reduce_loop(ndim),
        "    }\n"
        "};\n",
        generate_reduce_kernel_caller_code("ArgumentAllReduceKernel", "argument_all_reduce_kernel", ndim, 1)
    );
}

std::string create_axis_reduce_kernel_caller(int ndim) {
    std::string linear_index_operator_code("");
    if (ndim == 2) {
        linear_index_operator_code = utils::make_message(
            "    XINLINE T operator()(int i) const {\n"
            "        Shape<2> query;\n"
            "        query[0] = i;\n",
            construct_axis_reduce_for_loop(ndim),
            "    }\n"
        );
    }

    return utils::make_message(
        generate_reduce_kernel_template_code(1),
        "struct AxisReduceKernel", ndim, " {\n",
        generate_axis_reduce_kernel_constructor("AxisReduceKernel", ndim, ndim - 1),
        "    XINLINE T operator[](Shape<", ndim - 1, "> input_query) const {\n"
        "        Shape<", ndim, "> query = input_query.expand_dims(", ndim, ");\n",
        construct_axis_reduce_for_loop(ndim),
        "    }\n",
        linear_index_operator_code,
        "};\n",
        generate_reduce_kernel_caller_code("AxisReduceKernel", "axis_reduce_kernel", ndim, 1)
    );
}

std::string create_argument_axis_reduce_kernel_caller(int ndim) {
    std::string linear_index_operator_code("");
    if (ndim == 2) {
        linear_index_operator_code = utils::make_message(
            "    XINLINE T operator()(int i) const {\n"
            "        Shape<2> query;\n"
            "        query[0] = i;\n",
            construct_argument_axis_reduce_for_loop(ndim),
            "    }\n"
        );
    }

    return utils::make_message(
        generate_reduce_kernel_template_code(1),
        "struct ArgumentAxisReduceKernel", ndim, " {\n",
        generate_axis_reduce_kernel_constructor("ArgumentAxisReduceKernel", ndim, ndim - 1),
        "    XINLINE T operator[](Shape<", ndim - 1, "> input_query) const {\n"
        "        Shape<", ndim, "> query = input_query.expand_dims(", ndim, ");\n",
        construct_argument_axis_reduce_for_loop(ndim),
        "    }\n",
        linear_index_operator_code,
        "};\n",
        generate_reduce_kernel_caller_code("ArgumentAxisReduceKernel", "argument_axis_reduce_kernel", ndim, 1)
    );
}
