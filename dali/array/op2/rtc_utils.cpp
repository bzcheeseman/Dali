#include "rtc_utils.h"
#include "dali/utils/assert2.h"
#include "dali/utils/make_message.h"

// keeps rightmost (lowest) dimensions
std::string insert_auto_reshaped_variable(const std::string& name, int rank) {
    if (rank == 1) {
        return utils::make_message(
            name, ".ndim() == ", rank, " ? ",
            name, " : ", name, ".copyless_ravel()"
        );
    } else {
        return utils::make_message(
            name, ".copyless_right_fit_ndim(", rank, ")"
        );
    }
}


std::string build_views_constructor(
        const std::string& cpp_type,
        const std::vector<bool>& contiguous,
        int rank,
        int start_arg) {
    utils::MS stream;
    for (auto contig : contiguous) {
        stream << "    auto arg_" << start_arg
               << "_view = make"
               << (contig ? "_" : "_strided_")
               << "view<" << cpp_type << ", "
               << rank << ">("
               << insert_auto_reshaped_variable(
                    utils::make_message("arguments[", start_arg, "]"),
                    rank
                  )
               << ");\n";
        start_arg += 1;
    }
    return stream;
}

std::string build_view_constructor(const std::string& cpp_type,
                                   bool contiguous,
                                   int rank,
                                   const std::string& varname) {
    return utils::make_message(
        "    auto ", varname, "_view = make",
        (contiguous ? "_" : "_strided_"), "view<",
        cpp_type, ", ", rank, ">(",
        insert_auto_reshaped_variable(varname, rank),
        ");\n"
    );
}

std::string construct_for_loop(int rank, const std::string& code) {
    std::string for_loop = utils::make_message(
        "    Shape<", rank, "> query;\n"
    );
    for (int rank_num = 0; rank_num < rank; rank_num++) {
        std::string iname = "i" + std::to_string(rank_num);
        for_loop += utils::make_message(
            "    int& ", iname, " = query[", rank_num, "];\n"
        );
    }
    for (int rank_num = 0; rank_num < rank; rank_num++) {
        std::string iname = "i" + std::to_string(rank_num);
        for_loop += utils::make_message(
            std::string(4 + rank_num * 4, ' '),
            "for (", iname, " = 0; ", iname,
            " < dst_view.shape()[", rank_num, "]; ",
            iname, "++) {\n"
        );
    }
    for_loop += utils::make_message(
        std::string(4 + rank * 4, ' '),
        code
    );
    for (int rank_num = rank - 1; rank_num >= 0; rank_num--) {
        for_loop += utils::make_message(
            std::string(4 + rank_num * 4, ' '), "}\n"
        );
    }
    return for_loop;
}

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

std::vector<int> get_function_bshape(const std::vector<std::vector<int>>& bshapes) {
    if (bshapes.size() == 0) return {};

    std::vector<int> output_bshape = bshapes[0];
    for (size_t dim = 0; dim < output_bshape.size(); dim++) {
        for (size_t other_shape_idx = 1; other_shape_idx < bshapes.size(); other_shape_idx++) {
            ASSERT2(
                (output_bshape[dim] == bshapes[other_shape_idx][dim]) ||
                (output_bshape[dim] == -1 || bshapes[other_shape_idx][dim] == -1),
                "shapes dont match"
            );
            if (bshapes[other_shape_idx][dim] != -1) {
                output_bshape[dim] = bshapes[other_shape_idx][dim];
            }
        }
    }
    return output_bshape;
}

namespace {
    std::string generate_elementwise_kernel_class_signature(int num_args) {
        utils::MS template_data_stream;
        template_data_stream << "ElementWiseKernel" << num_args << "<Functor, ";
        for (int i = 0; i < num_args; i++) {
            template_data_stream << "C" << (i+1);
            if (i + 1 != num_args) {
                template_data_stream << ", ";
            } else {
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
        template_data_stream << "template<template <typename> class Functor,\n";
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

    std::string generate_elementwise_kernel_constructor_arguments(int num_args) {
        utils::MS template_data_stream;
        for (int i = 0; i < num_args; i++) {
            template_data_stream << "        const C" << (i+1) << "& "
                                 << generate_elementwise_kernel_argument(i);
            if (i + 1 != num_args) {
                template_data_stream << ",\n";
            } else {
                template_data_stream << ")";
            }
        }
        return template_data_stream;
    }

    std::string generate_elementwise_kernel_constructor(int num_args) {
        utils::MS template_data_stream;
        for (int i = 0; i < num_args; i++) {
            template_data_stream << "    const C" << (i+1) << "& "
                                 << generate_elementwise_kernel_argument(i) << "_;\n";
        }
        template_data_stream << "    static const int ndim = C1::ndim;\n"
                                "    typedef typename C1::T T;\n";
        template_data_stream << "    XINLINE ElementWiseKernel" << num_args << "(\n";
        template_data_stream << generate_elementwise_kernel_constructor_arguments(num_args) << "\n";
        template_data_stream << "        : ";
        for (int i = 0; i < num_args; i++) {
            template_data_stream << generate_elementwise_kernel_argument(i)
                                 << "_(" << generate_elementwise_kernel_argument(i) << ")";
            if (i + 1 != num_args) {
                template_data_stream << ", ";
            } else {
                template_data_stream << " {}\n";
            }
        }
        return template_data_stream;
    }

    std::string generate_elementwise_kernel_caller_code(int num_args) {
        return utils::make_message(
            generate_elementwise_kernel_template_code(num_args),
            "XINLINE ", generate_elementwise_kernel_class_signature(num_args), " element_wise_kernel(\n",
            generate_elementwise_kernel_constructor_arguments(num_args), " {\n",
            "    return ", generate_elementwise_kernel_class_signature(num_args), "(",
            generate_elementwise_kernel_arguments(num_args, ""), ");\n",
            "}\n"
        );
    }
}

std::string create_elementwise_kernel_caller(int num_args) {
    return utils::make_message(
        generate_elementwise_kernel_template_code(num_args),
        "struct ElementWiseKernel", num_args, " {\n",
        generate_elementwise_kernel_constructor(num_args),
        "    XINLINE T operator[](Shape<ndim> query) const {\n"
        "        return Functor<T>::Map(", generate_elementwise_kernel_arguments(num_args, "_[query]"), ");\n",
        "    }\n",
        "    XINLINE T operator()(int i) const {\n"
        "        return Functor<T>::Map(", generate_elementwise_kernel_arguments(num_args, "_(i)"), ");\n",
        "    }\n"
        "};\n",
        generate_elementwise_kernel_caller_code(num_args)
    );
}
