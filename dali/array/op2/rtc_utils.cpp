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

std::string build_scalar_constructor(const std::string& cpp_type,
                                     int rank,
                                     int start_arg) {
    return utils::make_message(
        "    auto scalar_", start_arg, " = make_scalar_view<",
        cpp_type, ", ", rank, ">(scalar_arguments[", start_arg, "]);\n"
    );
}

std::string construct_for_loop(int rank, const std::string& code, const std::string& varname, int indent) {
    std::string for_loop = utils::make_message(
        std::string(indent, ' '), "Shape<", rank, "> query;\n"
    );
    for (int rank_num = 0; rank_num < rank; rank_num++) {
        std::string iname = "i" + std::to_string(rank_num);
        for_loop += utils::make_message(
            std::string(indent, ' '), "int& ", iname, " = query[", rank_num, "];\n"
        );
    }
    for (int rank_num = 0; rank_num < rank; rank_num++) {
        std::string iname = "i" + std::to_string(rank_num);
        for_loop += utils::make_message(
            std::string(indent + rank_num * 4, ' '),
            "#pragma clang loop vectorize(enable)\n",
            std::string(indent + rank_num * 4, ' '),
            "#pragma clang loop interleave(enable)\n",
            std::string(indent + rank_num * 4, ' '),
            "for (", iname, " = 0; ", iname,
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
        for (const auto& other_bshape : bshapes) {
            if (other_bshape.size() == 0) continue;
            ASSERT2(other_bshape.size() == output_bshape.size(),
                "inputs must be scalars or have the same dimensionality."
            );
            ASSERT2(
                (output_bshape[dim] == other_bshape[dim]) ||
                (output_bshape[dim] == -1 || other_bshape[dim] == -1),
                "shapes dont match"
            );
            if (other_bshape[dim] != -1) {
                output_bshape[dim] = other_bshape[dim];
            }
        }
    }
    return output_bshape;
}

