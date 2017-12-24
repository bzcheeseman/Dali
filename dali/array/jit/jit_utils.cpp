#include "jit_utils.h"
#include "dali/utils/assert2.h"
#include "dali/utils/make_message.h"
#include "dali/utils/core_utils.h"

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

std::string build_array_definition(const std::string& cpp_type,
                                   const std::string& varname,
                                   bool contiguous,
                                   int rank,
                                   const std::string& constructor_arguments) {
    return utils::make_message(
        "    auto ", varname, " = make",
        (contiguous ? "_" : "_strided_"), "view<",
        cpp_type, ", ", rank, ">(", constructor_arguments, ");\n"
    );
}

std::string build_scalar_definition(const std::string& cpp_type,
                                    const std::string& varname,
                                    int rank,
                                    const std::string& captured_name) {
    return utils::make_message(
        "    auto ", varname, " = make_scalar_view<",
        cpp_type, ", ", rank, ">(", captured_name, ");\n"
    );
}

std::string build_shape_definition(const std::string& varname,
                                   int rank,
                                   const std::string& captured_name) {
    return utils::make_message(
        "    Shape<", rank, "> ", varname, "(", captured_name, ");\n"
    );
}

std::string generate_accessor_string(int rank) {
    std::stringstream ss;
    ss << "{";
    for (int i = 0; i < rank; ++i) {
        ss << "i" << i << (i +1 == rank ? "" : ",");
    }
    ss << "}";
    return ss.str();
}

std::string construct_for_loop(int rank, const std::string& code, const std::string& varname, int indent) {
    std::string for_loop;

    for (int rank_num = 0; rank_num < rank; rank_num++) {
        std::string iname = "i" + std::to_string(rank_num);
        for_loop += utils::make_message(
            std::string(indent + rank_num * 4, ' '),
            "#pragma clang loop vectorize(enable)\n",
            std::string(indent + rank_num * 4, ' '),
            "#pragma clang loop interleave(enable)\n",
            std::string(indent + rank_num * 4, ' '),
            "for (int ", iname, " = 0; ", iname,
            " < ", varname, ".shape()[", rank_num, "]; ",
            iname, "++) {\n"
        );
    }
    for_loop += utils::make_message(
        std::string(indent + rank * 4, ' '),
        code
    );
    for (int rank_num = rank - 1; rank_num >= 0; rank_num--) {
        for_loop += utils::make_message(
            std::string(indent + rank_num * 4, ' '), "}\n"
        );
    }
    return for_loop;
}

void ensure_output_array_compatible(const Array& out, const DType& output_dtype, const std::vector<int>& output_bshape) {
    if (out.is_stateless()) {
        return;
    }
    for (const int& dim_size: out.bshape()) {
        ASSERT2(dim_size >= -1,
            "Cannot assign to broadcasted output with broadcasted dimension"
            " bigger than 1, because it results in many-to-one mappings.");
    }

    bool output_bshape_compatible = out.ndim() == output_bshape.size();
    if (output_bshape_compatible) {
        for (int i = 0; i < out.ndim(); ++i) {
            if (output_bshape[i] != -1 && std::abs(output_bshape[i]) != out.shape()[i]) {
                output_bshape_compatible = false;
                break;
            }
        }
    }

    ASSERT2(output_bshape_compatible, utils::make_message("Cannot assign "
        "result of shape ", output_bshape, " to a location of shape ", out.shape(), "."));
    ASSERT2(out.dtype() == output_dtype, utils::make_message("Cannot assign "
        "result of dtype ", output_dtype, " to a location of dtype ", out.dtype(), "."));
}

std::vector<int> get_common_bshape(const std::vector<std::vector<int>>& bshapes) {
    if (bshapes.size() == 0) return {};

    int ndim_max = 0;
    int idx_max = 0;
    for (int idx = 0; idx < bshapes.size(); idx++) {
        if (bshapes[idx].size() > ndim_max) {
            ndim_max = bshapes[idx].size();
            idx_max = idx;
        }
    }
    std::vector<int> output_bshape = bshapes[idx_max];

    for (int dim = 0; dim < ndim_max; dim++) {
        for (const auto& other_bshape : bshapes) {
            if (other_bshape.size() == 0) continue;
            ASSERT2(other_bshape.size() == output_bshape.size(),
                "inputs must be scalars or have the same dimensionality."
            );
            ASSERT2(
                (output_bshape[dim] == other_bshape[dim]) ||
                (output_bshape[dim] == -1 || other_bshape[dim] == -1),
                utils::make_message(
                    "Could not find a common shape between ",
                    output_bshape, " and ", other_bshape, ".")
            );
            if (other_bshape[dim] != -1) {
                output_bshape[dim] = other_bshape[dim];
            }
        }
    }
    return output_bshape;
}

std::vector<int> get_common_bshape(const std::vector<Array>& arrays) {
    std::vector<std::vector<int>> arg_bshapes;
    for (const auto& array : arrays) {
        arg_bshapes.emplace_back(array.bshape());
    }
    return get_common_bshape(arg_bshapes);
}


    // start_(0) + indices_to_offset(shape_, query) * step_(0);\n"

std::string define_kernel(int ndim, bool has_shape,
                          const std::vector<std::string>& arguments,
                          std::string kernel, std::string kernel_name) {

    ASSERT2(kernel_name.size() > 0, "kernel_name must be a non-empty string.");
    ASSERT2(ndim > 0, utils::make_message("ndim must be strictly positive (got ndim=", ndim, ")."));
    size_t num_args = arguments.size();
    ASSERT2(num_args >= 0, utils::make_message("num_args must be >= 0 (got arguments.size()=", num_args, ")."));
    std::string shape_arg;
    if (has_shape) {
        shape_arg = utils::make_message("const Shape<", ndim, ">&");
    }
    std::string name = utils::make_message(char(std::toupper(kernel_name[0])),
                                           kernel_name.substr(1),
                                           "Kernel");

    std::string templated_caller;
    std::string templated_declarer;
    std::string call_arguments_definition;
    std::string call_arguments;
    std::string constructor_arguments;
    std::string member_variables;

    std::stringstream ss_caller;
    std::stringstream ss_declarer;
    std::stringstream ss_call_arguments_definition;
    std::stringstream ss_call_arguments;
    std::stringstream ss_constructor_arguments;
    std::stringstream ss_member_variables;
    if (num_args > 0) {
        ss_caller << "<";
        ss_declarer << "template<";
    }
    for (int i = 0; i < num_args; i++) {
        ss_caller << "C" << (i+1);
        ss_declarer << "typename C" << (i+1);
        ss_call_arguments_definition << "const C" << (i+1) << "& " << arguments[i];
        ss_call_arguments << arguments[i];
        ss_constructor_arguments << arguments[i] << "_(" << arguments[i] << ")";
        ss_member_variables << "    const C" << (i+1) << " " << arguments[i] << "_;\n";
        if (i + 1 != num_args) {
            ss_declarer << ", ";
            ss_caller << ", ";
            ss_call_arguments << ", ";
            ss_call_arguments_definition << ", ";
            ss_constructor_arguments << ", ";
        } else {
            ss_declarer << ">";
            ss_caller << ">";
        }
    }
    if (num_args > 0 & has_shape) {
        ss_call_arguments_definition << ", ";
        ss_call_arguments << ", ";
        ss_constructor_arguments << ", ";
    }
    templated_caller = ss_caller.str();
    templated_declarer = ss_declarer.str();
    if (has_shape) {
        ss_call_arguments_definition << "const Shape<" << ndim << ">& shape";
        ss_call_arguments << "shape";
        ss_constructor_arguments << "shape_(shape)";
        ss_member_variables << "    const Shape<" << ndim << "> shape_;\n";
    }
    call_arguments = ss_call_arguments.str();
    call_arguments_definition = ss_call_arguments_definition.str();
    constructor_arguments = ss_constructor_arguments.str();
    member_variables = ss_member_variables.str();

    std::string get_shape_fun;
    if (has_shape) {
        get_shape_fun = "    XINLINE const Shape<ndim>& shape() const {return shape_;}\n";
    }
    std::string typedefinition;
    if (num_args > 0) {
        typedefinition = "    typedef typename C1::T T;\n";
    }

    std::string kernel_tail;
    if (utils::endswith(kernel, ";\n")) {
        kernel_tail = "";
    } else {
        if (utils::endswith(kernel, ";")) {
            kernel_tail = "\n";
        } else {
            kernel_tail = ";\n";
        }
    }

    return utils::make_message(templated_declarer, "\n",
        "struct ", name, " {\n", member_variables,
        "    static const int ndim = ", ndim, ";\n", typedefinition, get_shape_fun,
        "    XINLINE ", name, "(", call_arguments_definition, ")"
        "       : ", constructor_arguments, " {}\n"
        "    XINLINE T operator[](const Shape<ndim>& query) const {\n"
        "        return ", kernel, kernel_tail,
        "    }\n"
        "};\n", templated_declarer, "\n",
        name, templated_caller, " ", kernel_name,
        "(", call_arguments_definition, ") {\n"
        "    return ", name, templated_caller, "(", call_arguments, ");\n"
        "}\n"
    );
}
