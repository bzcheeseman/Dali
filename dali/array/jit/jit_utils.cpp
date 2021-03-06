#include "jit_utils.h"

#include <algorithm>

#include "dali/utils/assert2.h"
#include "dali/utils/make_message.h"
#include "dali/utils/core_utils.h"

namespace {
    std::string add_if_missing(std::string code, const std::string& missing) {
        if (utils::endswith(code, missing)) {
            return code;
        } else {
            return utils::make_message(code, missing);
        }
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

std::string construct_for_loop(int rank, std::string code, const std::string& varname, int indent) {
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
    code = add_if_missing(code, ";\n");
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

void ensure_output_array_compatible(const Array& out, const DType& output_dtype, const std::vector<int>& output_shape) {
    if (out.is_stateless()) {
        return;
    }
    bool output_shape_compatible = out.ndim() == output_shape.size();
    if (output_shape_compatible) {
        for (int i = 0; i < out.ndim(); ++i) {
            if (output_shape[i] != 1 && output_shape[i] != out.shape()[i]) {
                output_shape_compatible = false;
                break;
            }
        }
    }
    ASSERT2(output_shape_compatible, utils::make_message("Cannot assign "
        "result of shape ", output_shape, " to a location of shape ", out.shape(), "."));
    ASSERT2(out.dtype() == output_dtype, utils::make_message("Cannot assign "
        "result of dtype ", output_dtype, " to a location of dtype ", out.dtype(), "."));
}

std::vector<int> get_common_shape(const std::vector<const std::vector<int>*>& shapes) {
    if (shapes.size() == 0) return {};

    int ndim_max = 0;
    int idx_max = 0;
    for (int idx = 0; idx < shapes.size(); idx++) {
        if (shapes[idx]->size() > ndim_max) {
            ndim_max = shapes[idx]->size();
            idx_max = idx;
        }
    }
    std::vector<int> output_shape = *shapes[idx_max];

    for (int dim = 0; dim < ndim_max; dim++) {
        for (auto other_shape : shapes) {
            if (other_shape->size() == 0) continue;
            ASSERT2(other_shape->size() == output_shape.size(),
                "inputs must be scalars or have the same dimensionality."
            );
            ASSERT2(
                (output_shape[dim] == (*other_shape)[dim]) ||
                (output_shape[dim] == 1 || (*other_shape)[dim] == 1),
                utils::make_message(
                    "Could not find a common shape between ",
                    output_shape, " and ", (*other_shape), ".")
            );
            if ((*other_shape)[dim] != 1) {
                output_shape[dim] = (*other_shape)[dim];
            }
        }
    }
    return output_shape;
}

std::vector<int> get_common_shape(const std::vector<Array>& arrays) {
    std::vector<const std::vector<int>*> arg_shapes;
    for (const auto& array : arrays) {
        arg_shapes.emplace_back(&array.shape());
    }
    return get_common_shape(arg_shapes);
}

void define_kernel(int ndim, bool has_shape,
                   const std::vector<std::string>& arguments,
                   std::string kernel, std::string kernel_name,
                   bool assignment_code,
                   op::jit::insert_t insert) {
    ASSERT2(kernel_name.size() > 0, "kernel_name must be a non-empty string.");
    ASSERT2(ndim > 0, utils::make_message("ndim must be strictly positive (got ndim=", ndim, ")."));
    size_t num_args = arguments.size();
    ASSERT2(num_args >= 0, utils::make_message("num_args must be >= 0 (got arguments.size()=", num_args, ")."));
    std::string shape_arg;
    if (has_shape) {
        shape_arg = utils::make_message("const Shape<", ndim, ">&");
    }
    std::string name = utils::make_message(char(std::toupper(kernel_name[0])),
                                           kernel_name.substr(1),
                                           assignment_code ? "Assign" : "",
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
        if (assignment_code) {
            ss_member_variables << "    C" << (i+1) << " " << arguments[i] << "_;\n";
        } else {
            ss_member_variables << "    const C" << (i+1) << " " << arguments[i] << "_;\n";
        }
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
    if ((num_args > 0) & has_shape) {
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

    kernel = utils::trim(kernel);
    if (!utils::endswith(kernel, ";\n") && !utils::endswith(kernel, ";")) {
        kernel = utils::make_message(kernel, ";");
    }
    if (kernel.find("return") == std::string::npos) {
        kernel = utils::make_message("return ", kernel);
    }
    std::stringstream ss_kernel;
    int stack_depth = 2, new_stack_depth;
    for (const auto& line : utils::split(kernel, '\n', false)) {
        new_stack_depth = stack_depth + std::count(line.begin(), line.end(), '{') - std::count(line.begin(), line.end(), '}');
        if (new_stack_depth == stack_depth - 1) {
            ss_kernel << std::string(new_stack_depth * 4, ' ');
        } else if (new_stack_depth == stack_depth + 1) {
            ss_kernel << std::string(stack_depth * 4, ' ');
        } else {
            ss_kernel << std::string(new_stack_depth * 4, ' ');
        }
        stack_depth = new_stack_depth;
        ss_kernel << utils::trim(line) << "\n";
    }
    kernel = ss_kernel.str();

    std::string return_type;
    if (assignment_code) {
        return_type = "const T&";
    } else {
        return_type = "T";
    }
    std::string const_query;
    std::string modifiable_query;
    if (assignment_code) {
        const_query = utils::make_message(
            "    XINLINE const T& operator[](const Shape<ndim>& query) const {\n",
            kernel,
            "    }\n");
        modifiable_query = utils::make_message(
            "    XINLINE T& operator[](const Shape<ndim>& query) {\n",
            kernel,
            "    }\n");
    } else {
        const_query = utils::make_message(
        "    XINLINE T operator[](const Shape<ndim>& query) const {\n",
        kernel,
        "    }\n");
    }

    insert(utils::make_message(templated_declarer, "\n",
        "struct ", name, " {\n", member_variables,
        "    static const int ndim = ", ndim, ";\n", typedefinition, get_shape_fun,
        "    XINLINE ", name, "(", call_arguments_definition, ")\n"
        "       : ", constructor_arguments, " {}\n",
        const_query, modifiable_query,
        "};\n"));
    insert(utils::make_message(templated_declarer, "\n",
        name, templated_caller, " ", kernel_name,
        "(", call_arguments_definition, ") {\n"
        "    return ", name, templated_caller, "(", call_arguments, ");\n"
        "}\n"));
}

std::string generate_call_code_nd(const Expression* expr,
                                  const std::string& kernel_name,
                                  const op::jit::SymbolTable& symbol_table,
                                  memory::DeviceT device_type,
                                  bool has_shape) {
    std::stringstream ss;
    ss << kernel_name << "(";
    const auto& args = expr->arguments();
    for (size_t i = 0; i < args.size(); i++) {
        ss << op::jit::get_call_code_nd(args[i], symbol_table, device_type);
        if (i + 1 != args.size()) {
            ss << ", ";
        }
    }
    if (has_shape) {
        if (args.size() > 0) {
            ss << ", ";
        }
        ss << symbol_table.get_shape(expr);
    }
    ss << ")";
    return ss.str();
}
