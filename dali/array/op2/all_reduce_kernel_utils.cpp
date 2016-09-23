#include "elementwise_kernel_utils.h"
#include "dali/utils/make_message.h"
#include "dali/utils/print_utils.h"
#include "dali/array/op2/rtc_utils.h"

namespace {
    std::string generate_all_reduce_kernel_class_signature(int ndim) {
        return utils::make_message("AllReduceKernel", ndim, "<Reducer, Type, C1>");
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

    std::string generate_all_reduce_kernel_constructor(int ndim, int result_ndim) {
        auto arg = generate_all_reduce_kernel_argument();
        return utils::make_message(
            "    const C1& ", arg, "_;\n"
            "    static const int ndim = ", result_ndim, ";\n"
            "    typedef Type T;\n"
            "    XINLINE const Shape<ndim>& shape() const {\n"
            "        return ", generate_all_reduce_kernel_argument(), "_.shape();\n"
            "    }\n"
            "    XINLINE AllReduceKernel", ndim, "(\n",
            generate_all_reduce_kernel_constructor_arguments(), "\n"
            "        : ", arg, "_(", arg, ") {}\n"
        );
    }

    std::string generate_all_reduce_kernel_caller_code(int ndim) {
        return utils::make_message(
            generate_all_reduce_kernel_template_code(),
            "XINLINE ", generate_all_reduce_kernel_class_signature(ndim), " all_reduce_kernel_", ndim, "d(\n",
            generate_all_reduce_kernel_constructor_arguments(), " {\n",
            "    return ", generate_all_reduce_kernel_class_signature(ndim), "(",
            generate_all_reduce_kernel_argument(), ");\n",
            "}\n"
        );
    }
}

// For reference:
// expr::Plan<R, DType> dplan = MakePlan(dst->self());
//   expr::Plan<E, DType> splan = MakePlan(exp.self());
//   for (index_t c = 0; c < pshape[1]; ++c) {
//     DType res; Reducer::SetInitValue(res);
//     for (index_t n = 0; n < pshape[0]; ++n) {
//       DType tres; Reducer::SetInitValue(tres);
//       for (index_t y = 0; y < pshape[2]; ++y) {
//         for (index_t x = 0; x < pshape[3]; ++x) {
//           Reducer::Reduce(tres,
//                           splan.Eval((n * pshape[1] + c) * pshape[2] + y, x));
//         }
//       }
//       Reducer::Reduce(res, tres);
//     }
//     Saver::Save(dplan.REval(0, c), DType(res * scale));
//   }

// T res;
// Reducer<T>::SetInitValue(res);
// Shape<ndim> query;
// int& i = query[0];
// int& j = query[1];
// for (int i = 0; i < ....; i++) {
//     for (int j = 0; j < ...; j++) {
//         Reducer::Reduce(res, tres[query]);
//     }
// }
// return res;

std::string create_all_reduce_kernel_caller(int ndim, int result_ndim) {
    return utils::make_message(
        generate_all_reduce_kernel_template_code(),
        "struct AllReduceKernel", ndim, " {\n",
        generate_all_reduce_kernel_constructor(ndim, result_ndim),
        "    XINLINE T operator[](Shape<", result_ndim, ">) const {\n"
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
        "    }\n"
        "    XINLINE T operator()(int) const {\n"
        "        int num_el = ", generate_all_reduce_kernel_argument(), "_.shape()[0];\n"
        "        T res;\n"
        "        Reducer::SetInitValue(res);\n"
        "        #pragma clang loop vectorize(enable)\n"
        "        #pragma clang loop interleave(enable)\n"
        "        for (int i = 0; i < num_el; ++i) {\n"
        "            Reducer::Reduce(res, ", generate_all_reduce_kernel_argument(), "_(i));\n"
        "        }\n"
        "        return res;\n"
        "    }\n"
        "};\n",
        generate_all_reduce_kernel_caller_code(ndim)
    );
}
