#include "elementwise_kernel_utils.h"
#include "dali/utils/make_message.h"
#include "dali/utils/print_utils.h"

namespace {
    std::string kernel_struct_name(int num_args, int ndim) {
        return utils::make_message("ElementWiseKernel", ndim, "D", num_args);
    }

    std::string generate_elementwise_kernel_class_signature(int num_args, int ndim) {
        utils::MS template_data_stream;
        template_data_stream << kernel_struct_name(num_args, ndim) << "<Functor, Type, ";
        for (int i = 0; i < num_args; i++) {
            template_data_stream << "C" << (i+1);
            if (i + 1 != num_args) {
                template_data_stream << ", ";
            } else {
                template_data_stream << ">";
            }
        }
        return template_data_stream;
    }

    std::string generate_elementwise_kernel_argument(int arg) {
        return utils::make_message("arg_", arg + 1, "_view");
    }

    std::string generate_elementwise_kernel_arguments(int num_args, const std::string& query_code) {
        utils::MS template_data_stream;
        for (int i = 0; i < num_args; i++) {
            template_data_stream << generate_elementwise_kernel_argument(i) << query_code;
            if (i + 1 != num_args) {
                template_data_stream << ", ";
            }
        }
        return template_data_stream;
    }

    std::string generate_elementwise_kernel_template_code(int num_args) {
        utils::MS template_data_stream;
        template_data_stream << "template<template <typename> class Functor, typename Type,\n";
        for (int i = 0; i < num_args; i++) {
            template_data_stream << "         typename C" << (i + 1);
            if (i + 1 != num_args) {
                template_data_stream << ",\n";
            } else {
                template_data_stream << ">\n";
            }
        }
        return template_data_stream;
    }

    std::string generate_elementwise_kernel_constructor_arguments(int num_args, int ndim) {
        utils::MS template_data_stream;
        for (int i = 0; i < num_args; i++) {
            template_data_stream << "        const C" << (i+1) << "& "
                                 << generate_elementwise_kernel_argument(i);
            if (i + 1 != num_args) {
                template_data_stream << ",\n";
            } else {
                template_data_stream << ")";
            }
        }
        return template_data_stream;
    }

    std::string generate_elementwise_kernel_constructor(int num_args, int ndim) {
        utils::MS template_data_stream;
        for (int i = 0; i < num_args; i++) {
            template_data_stream << "    C" << (i+1) << " "
                                 << generate_elementwise_kernel_argument(i) << "_;\n";
        }
        template_data_stream << "    static const int ndim = " << ndim << ";\n"
                             << "    typedef Type T;\n"
                             << "    XINLINE const Shape<ndim>& shape() const {\n"
                             << "        return arg_1_view_.shape();\n"
                             << "    }\n"
                             << "    XINLINE " << kernel_struct_name(num_args, ndim) << "(\n"
                             << generate_elementwise_kernel_constructor_arguments(num_args, ndim) << "\n"
                             << "        : ";
        for (int i = 0; i < num_args; i++) {
            template_data_stream << generate_elementwise_kernel_argument(i)
                                 << "_(" << generate_elementwise_kernel_argument(i) << ")";
            if (i + 1 != num_args) {
                template_data_stream << ", ";
            }
        }
        template_data_stream << " {}\n";
        return template_data_stream;
    }

    std::string generate_elementwise_kernel_caller_code(int num_args, int ndim) {
        return utils::make_message(
            generate_elementwise_kernel_template_code(num_args),
            "XINLINE ", generate_elementwise_kernel_class_signature(num_args, ndim), " ", elementwise_kernel_name(num_args, ndim), "(\n",
            generate_elementwise_kernel_constructor_arguments(num_args, ndim), " {\n",
            "    return ", generate_elementwise_kernel_class_signature(num_args, ndim), "(",
            generate_elementwise_kernel_arguments(num_args, ""), ");\n",
            "}\n"
        );
    }
}

std::string elementwise_kernel_name(int num_args, int ndim) {
    return utils::make_message("element_wise_kernel", ndim, "D", num_args);
}

void create_elementwise_kernel_caller(int num_args, int ndim,
                                      std::function<void(const std::string&)> insert) {
    insert(utils::make_message(
        generate_elementwise_kernel_template_code(num_args),
        "struct ", kernel_struct_name(num_args, ndim), " {\n",
        generate_elementwise_kernel_constructor(num_args, ndim),
        "    XINLINE T operator[](const Shape<ndim>& query) const {\n"
        "        return Functor<T>::Map(", generate_elementwise_kernel_arguments(num_args, "_[query]"), ");\n",
        "    }\n"
        "};\n"));
    insert(generate_elementwise_kernel_caller_code(num_args, ndim));
}
