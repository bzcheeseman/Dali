#include "fused_operation.h"

#include <unordered_set>

#include "dali/array/array.h"
#include "dali/array/function2/compiler.h"
#include "dali/array/op2/rtc_utils.h"
#include "dali/utils/tuple_hash.h"
#include "dali/utils/make_message.h"

FusedOperation::FusedOperation(const Array& arr) : type_(FUSED_OP_ARRAY_T), arr_(arr) {}
FusedOperation::FusedOperation(Array&& arr) : type_(FUSED_OP_ARRAY_T), arr_(arr) {}
FusedOperation::FusedOperation(const Assignable<Array>& arr) : type_(FUSED_OP_ARRAY_T), arr_(arr) {}
FusedOperation::FusedOperation(const double& scalar) : type_(FUSED_OP_SCALAR_T), scalar_(scalar) {}
FusedOperation::FusedOperation(FUSED_OP_T type,
                               const std::string& functor_name,
                               const std::vector<FusedOperation>& arguments) :
        type_(type), arguments_(arguments), functor_name_(functor_name) {}
FusedOperation::FusedOperation(FUSED_OP_T type,
                               const std::string& functor_name,
                               const std::string& extra_code,
                               const std::vector<FusedOperation>& arguments) :
        type_(type), arguments_(arguments), functor_name_(functor_name), extra_code_(extra_code) {}

std::vector<Array> FusedOperation::get_arrays() const {
    std::vector<Array> out;
    get_arrays(&out);
    return out;
}

void FusedOperation::get_arrays(std::vector<Array>* arrs) const {
    if (type_ == FUSED_OP_ARRAY_T) {
        arrs->emplace_back(arr_);
    } else {
        for (auto& arg : arguments_) {
            arg.get_arrays(arrs);
        }
    }
}

std::vector<double> FusedOperation::get_scalars() const {
    std::vector<double> out;
    get_scalars(&out);
    return out;
}

void FusedOperation::get_scalars(std::vector<double>* arrs) const {
    if (type_ == FUSED_OP_SCALAR_T) {
        arrs->emplace_back(scalar_);
    } else {
        for (auto& arg : arguments_) {
            arg.get_scalars(arrs);
        }
    }
}

int FusedOperation::ndim() const {
    if (type_ == FUSED_OP_ARRAY_T) {
        return arr_.ndim();
    } else if (type_ == FUSED_OP_SCALAR_T) {
        return 0;
    } else {
        return arguments_[0].ndim();
    }
}

DType FusedOperation::dtype() const {
    if (type_ == FUSED_OP_ARRAY_T) {
        return arr_.dtype();
    } else if (type_ == FUSED_OP_SCALAR_T) {
        return DTYPE_DOUBLE;
    } else {
        return arguments_[0].dtype();
    }
}

const FusedOperation::FUSED_OP_T& FusedOperation::type() const {
    return type_;
}
const Array& FusedOperation::array() const {
    return arr_;
}
const std::string& FusedOperation::functor_name() const {
    return functor_name_;
}
const std::string& FusedOperation::extra_code() const {
    return extra_code_;
}
const std::vector<FusedOperation>& FusedOperation::arguments() const {
    return arguments_;
}

std::vector<int> FusedOperation::bshape() const {
    if (type_ == FUSED_OP_ARRAY_T) {
        return arr_.bshape();
    } else if (type_ == FUSED_OP_SCALAR_T) {
        return {};
    } else {
        std::vector<std::vector<int>> bshapes;
        for (const auto& el : arguments_) {
            bshapes.emplace_back(el.bshape());
        }
        return get_function_bshape(bshapes);
    }
}

hash_t get_operation_hash(const FusedOperation& op) {
    if (op.type() == FusedOperation::FUSED_OP_ARRAY_T) {
        return utils::get_hash(
            std::make_tuple(
                op.type(),
                op.array().strides().empty()
            )
        );
    } else if (op.type() == FusedOperation::FUSED_OP_SCALAR_T) {
        return utils::get_hash(
            std::make_tuple(
                op.type(),
                true
            )
        );
    } else {
        std::vector<hash_t> hashes;
        hash_t arg_hash = 0;
        for (const auto& var : op.arguments()) {
            arg_hash ^= get_operation_hash(var) + 0x9e3779b9 + (arg_hash<<6) + (arg_hash>>2);
        }
        return utils::get_hash(
            std::make_tuple(
                op.type(),
                op.functor_name(),
                arg_hash
            )
        );
    }
}

int FusedOperation::type_to_min_rank(FUSED_OP_T type) {
    if (type == FUSED_OP_KERNEL_T) return 2;
    return 1;
}

int FusedOperation::computation_rank() const {
    if (type_ == 0) {
        if (arr_.strides().empty()) {
            return 1;
        } else {
            return arr_.ndim();
        }
    } else {
        int rank = FusedOperation::type_to_min_rank(type_);
        for (auto& arg : arguments_) {
            rank = std::max(rank, arg.computation_rank());
        }
        return rank;
    }
}

std::string FusedOperation::get_code_setup(const std::string& cpp_type, memory::Device device, int rank, int& arg_idx, int& scalar_arg_idx) const {
    if (type_ == FUSED_OP_ARRAY_T) {
        auto res = build_views_constructor(cpp_type, {arr_.strides().empty()}, rank, arg_idx);
        arg_idx += 1;
        return res;
    } else if (type_ == FUSED_OP_SCALAR_T) {
        auto res = build_scalar_constructor(cpp_type, rank, scalar_arg_idx);
        scalar_arg_idx += 1;
        return res;
    } else {
        utils::MS stream;
        for (auto& arg : arguments_) {
            stream << arg.get_code_setup(cpp_type, device, rank, arg_idx, scalar_arg_idx);
        }
        return stream;
    }
}

std::string FusedOperation::get_code_setup(const std::string& cpp_type, memory::Device device, int rank) const {
    int arg_idx = 0;
    int scalar_arg_idx = 0;
    return get_code_setup(cpp_type, device, rank, arg_idx, scalar_arg_idx);
}

std::string FusedOperation::get_call_nd(int rank) const {
    if (rank == 1) {
        return "(i)";
    } else {
        return "[query]";
    }
}

std::string FusedOperation::get_call_code_nd(const std::string& cpp_type, int& arg_idx, int& scalar_arg_idx) const {
    if (type_ == FUSED_OP_ARRAY_T) {
        auto res = utils::make_message("arg_", arg_idx, "_view");
        arg_idx += 1;
        return res;
    } else if (type_ == FUSED_OP_SCALAR_T) {
        auto res = utils::make_message("scalar_", scalar_arg_idx);
        scalar_arg_idx += 1;
        return res;
    } else if (type_ == FUSED_OP_KERNEL_T || type_ == FUSED_OP_ELEMENTWISE_T) {
        utils::MS stream;
        if (type_ == FUSED_OP_ELEMENTWISE_T) {
            stream << "element_wise_kernel<" << functor_name_ << ">(";
        } else if (type_ == FUSED_OP_KERNEL_T) {
            stream << functor_name_ << "(";
        }
        int args_called = 0;
        for (auto& arg : arguments_) {
            stream << arg.get_call_code_nd(cpp_type, arg_idx, scalar_arg_idx);
            args_called += 1;
            if (args_called != arguments_.size()) {
                stream << ", ";
            }
        }
        stream << ")";
        return stream;
    } else {
        ASSERT2(false, utils::MS() << "Unknown operation type (" << type_ << "). Use 0, 1, 2 or 3.");
    }
}

std::string FusedOperation::get_call_code_nd(const std::string& cpp_type) const {
    int arg_idx = 0;
    int scalar_arg_idx = 0;
    return get_call_code_nd(cpp_type, arg_idx, scalar_arg_idx);
}

std::string FusedOperation::get_assign_code_nd(const OPERATOR_T& operator_t, const std::string& cpp_type, const std::string& call_nd) const {
    return utils::make_message(
        "dst_view", call_nd, " ",
        operator_to_name(operator_t),
        " ", get_call_code_nd(cpp_type), call_nd, ";\n"
    );
}

class GeneratedCodeTracker {
    private:
        std::unordered_set<int> elementwise_kernels_;

    public:
        bool is_elementwise_kernel_generated(int size) const {
            return elementwise_kernels_.find(size) != elementwise_kernels_.end();
        }
        void mark_elementwise_kernel_generated(int size) {
            elementwise_kernels_.insert(size);
        }
};

void fused_operation_get_extra_code(const FusedOperation& fop,
                                    GeneratedCodeTracker* tracker,
                                    std::string* extra_code_ptr) {
    if (!fop.extra_code().empty()) {
        *extra_code_ptr = (*extra_code_ptr) + fop.extra_code();
    }
    if (fop.type() == FusedOperation::FUSED_OP_ELEMENTWISE_T && !tracker->is_elementwise_kernel_generated(fop.arguments().size())) {
        (*extra_code_ptr) = (
            (*extra_code_ptr) +
            create_elementwise_kernel_caller(fop.arguments().size())
        );
        tracker->mark_elementwise_kernel_generated(fop.arguments().size());
    }
    for (const auto& arg : fop.arguments()) {
        fused_operation_get_extra_code(arg, tracker, extra_code_ptr);
    }
}

std::string FusedOperation::get_extra_code() const {
    std::string result;
    GeneratedCodeTracker tracker;
    fused_operation_get_extra_code(*this, &tracker, &result);
    return result;
}


std::string FusedOperation::get_code_template(const OPERATOR_T& operator_t, bool dst_contiguous, DType dtype, memory::Device device, int rank) const {
    auto cpp_type = dtype_to_cpp_name(dtype);
    std::string code = utils::make_message(
        get_extra_code(),
        "void run(Array& dst, const std::vector<Array>& arguments, const std::vector<double>& scalar_arguments) {\n"
    );
    code += get_code_setup(cpp_type, device, rank);
    code += build_view_constructor(
        cpp_type, dst_contiguous, rank, "dst"
    );
    // now we declare output.
    std::string for_loop;
    std::string call_nd = get_call_nd(rank);
    if (rank == 1) {
        for_loop = utils::make_message(
            "    int num_el = dst.number_of_elements();\n"
            "    #pragma clang loop vectorize(enable)\n"
            "    #pragma clang loop interleave(enable)\n"
            "    for (int i = 0; i < num_el; ++i) {\n",
            "        ", get_assign_code_nd(operator_t, cpp_type, call_nd),
            "    }\n"
        );
    } else {
        for_loop = construct_for_loop(
            rank,
            get_assign_code_nd(operator_t, cpp_type, call_nd)
        );
    }
    code += for_loop;
    code += "}\n";
    return code;
}

std::function<void(Array&, const std::vector<Array>&, const std::vector<double>&)> FusedOperation::compile(
        const OPERATOR_T& operator_t, bool dst_contiguous, DType dtype, memory::Device device) const {
    // get the lowest dimension that suffices to compute this problem:
    int comp_rank = computation_rank();
    // given the lowest rank for which the operation can be executed
    // now check if the destination demands a higher rank (due to striding)
    if (!dst_contiguous) {
        comp_rank = std::max(comp_rank, ndim());
    }
    // compute a quasi-unique hash for the fused operation
    hash_t hash = utils::get_hash(
        std::make_tuple(
            dtype,
            operator_t,
            device.is_cpu(),
            get_operation_hash(*this),
            dst_contiguous,
            comp_rank
        )
    );
    // check if the operation needs to be runtime compiled
    if (!array_op_compiler.load(hash)) {
        auto code_template = get_code_template(
            operator_t,
            dst_contiguous,
            dtype,
            device,
            comp_rank
        );
        array_op_compiler.compile<Array&, const std::vector<Array>&, const std::vector<double>&>(
            hash,
            code_template,
            {}
        );
    }
    // return the operation that was loaded or compiled:
    return array_op_compiler.get_function<Array&, const std::vector<Array>&, const std::vector<double>&>(hash);
}

FusedOperation::operator Assignable<Array> () const {
    return Assignable<Array>([fop = *this](Array& out, const OPERATOR_T& operator_t) mutable {
        auto output_dtype = fop.dtype();
        auto output_device = memory::Device::cpu();
        auto output_bshape = fop.bshape();

        initialize_output_array(
            out,
            output_dtype,
            output_device,
            &output_bshape
        );

        bool dst_contiguous = out.strides().empty();
        auto fptr = fop.compile(operator_t, dst_contiguous, output_dtype, output_device);

        auto arrays = fop.get_arrays();
        auto scalars = fop.get_scalars();
        std::vector<Array> arrays_reshaped;

        for (auto& arr : arrays) {
            arrays_reshaped.emplace_back(
                arr.reshape_broadcasted(output_bshape)
            );
        }

        fptr(out, arrays_reshaped, scalars);
    });
}

std::size_t std::hash<FusedOperation::FUSED_OP_T>::operator()(const FusedOperation::FUSED_OP_T& k) const {
    return std::hash<int>()(k);
}

bool FusedOperation::dtype_compatible(const FusedOperation& a, const FusedOperation& b) {
    return (
        a.type_ == FUSED_OP_SCALAR_T ||
        b.type_ == FUSED_OP_SCALAR_T ||
        a.dtype() == b.dtype()
    );
}

bool FusedOperation::ndim_compatible(const FusedOperation& a, const FusedOperation& b) {
    return (
        a.type_ == FUSED_OP_SCALAR_T ||
        b.type_ == FUSED_OP_SCALAR_T ||
        a.ndim() == b.ndim()
    );
}

std::vector<int> get_function_bshape(const FusedOperation& a, const FusedOperation& b) {
    if (a.type() == FusedOperation::FUSED_OP_SCALAR_T) {
        return b.bshape();
    } else if (b.type() == FusedOperation::FUSED_OP_SCALAR_T) {
        return a.bshape();
    } else {
        return get_function_bshape({a.bshape(), b.bshape()});
    }
}

namespace op2 {
    FusedOperation elementwise(
        const FusedOperation& a,
        const std::string& functor_name) {
        return FusedOperation(
            FusedOperation::FUSED_OP_ELEMENTWISE_T,
            functor_name,
            {a}
        );
    }

    FusedOperation elementwise(
        const FusedOperation& a,
        const FusedOperation& b,
        const std::string& functor_name) {
        ASSERT2(FusedOperation::dtype_compatible(a, b), "dtypes don't match");
        ASSERT2(FusedOperation::ndim_compatible(a, b), "ranks don't match");
        auto output_bshape = get_function_bshape(a, b);

        return FusedOperation(
            FusedOperation::FUSED_OP_ELEMENTWISE_T,
            functor_name,
            {a, b}
        );
    }

    FusedOperation binary_kernel_function(
        const FusedOperation& a,
        const FusedOperation& b,
        const std::string& function_name,
        const std::string& kernel_code
    ) {
        ASSERT2(FusedOperation::dtype_compatible(a, b), "dtypes don't match");
        ASSERT2(FusedOperation::ndim_compatible(a, b), "ranks don't match");
        auto output_bshape = get_function_bshape(a, b);

        return FusedOperation(
            FusedOperation::FUSED_OP_KERNEL_T,
            function_name,
            kernel_code,
            {a, b}
        );
    }
}

