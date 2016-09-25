#include "fused_operation.h"

#include <unordered_set>

#include "dali/array/array.h"
#include "dali/array/function2/compiler.h"
#include "dali/array/op2/rtc_utils.h"
#include "dali/utils/tuple_hash.h"
#include "dali/utils/make_message.h"

#include "dali/array/op2/elementwise_kernel_utils.h"
#include "dali/array/op2/all_reduce_kernel_utils.h"

FusedOperation::FusedOperation(const Array& arr) : type_(FUSED_OP_ARRAY_T), arr_(arr), dtype_(arr.dtype()) {}
FusedOperation::FusedOperation(Array&& arr) : type_(FUSED_OP_ARRAY_T), arr_(arr), dtype_(arr.dtype()) {}
FusedOperation::FusedOperation(const Assignable<Array>& arr) : FusedOperation(Array(arr)) {}
FusedOperation::FusedOperation(const double& scalar) : type_(FUSED_OP_SCALAR_T), scalar_(scalar), dtype_(DTYPE_DOUBLE) {}
FusedOperation::FusedOperation(const int& scalar) : FusedOperation(op2::astype(FusedOperation(double(scalar)), DTYPE_INT32)) {}
FusedOperation::FusedOperation(FUSED_OP_T type,
                               const std::string& functor_name,
                               const std::vector<FusedOperation>& arguments,
                               DType dtype) :
        type_(type), arguments_(arguments), functor_name_(functor_name), dtype_(dtype) {}
FusedOperation::FusedOperation(FUSED_OP_T type,
                               const std::string& functor_name,
                               const std::string& extra_code,
                               const std::vector<FusedOperation>& arguments,
                               DType dtype) :
        type_(type), arguments_(arguments), functor_name_(functor_name), extra_code_(extra_code), dtype_(dtype) {}

std::vector<Array> FusedOperation::get_arrays(const std::vector<int>& bshape, const int& rank) const {
    std::vector<Array> out;
    get_arrays(&out, bshape, rank);
    return out;
}


void FusedOperation::get_arrays(std::vector<Array>* arrs, const std::vector<int>& bshape, const int& rank) const {
    switch (type_) {
        case FUSED_OP_ARRAY_T:
            if (rank == arr_.ndim()) {
                arrs->emplace_back(arr_.reshape_broadcasted(bshape));
            } else if (rank == 1) {
                arrs->emplace_back(arr_.reshape_broadcasted(bshape).copyless_ravel());
            } else {
                arrs->emplace_back(arr_.reshape_broadcasted(bshape).copyless_right_fit_ndim(rank));
            }
            break;
        case FUSED_OP_ALLREDUCE_T:
        case FUSED_OP_ARGUMENT_ALLREDUCE_T:
            for (auto& arg : arguments_) {
                arg.get_arrays(arrs, arg.shape(), arg.computation_rank());
            }
            break;
        case FUSED_OP_AXISREDUCE_LOW_DIM_T:
        case FUSED_OP_ARGUMENT_AXISREDUCE_LOW_DIM_T:
            for (auto& arg : arguments_) {
                arg.get_arrays(
                    arrs,
                    arg.shape(),
                    rank + 1
                );
            }
            break;
        default:
            for (auto& arg : arguments_) {
                arg.get_arrays(arrs, bshape, rank);
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
    switch (type_) {
        case FUSED_OP_ARRAY_T:
            return arr_.ndim();
        case FUSED_OP_SCALAR_T:
        case FUSED_OP_ALLREDUCE_T:
        case FUSED_OP_ARGUMENT_ALLREDUCE_T:
            return 0;
        case FUSED_OP_AXISREDUCE_LOW_DIM_T:
        case FUSED_OP_ARGUMENT_AXISREDUCE_LOW_DIM_T:
            return arguments_[0].ndim() - 1;
        default:
            return arguments_[0].ndim();
    }
}

bool FusedOperation::is_scalar() const {
    return ndim() == 0;
}

DType FusedOperation::dtype() const {
    return dtype_;
}
const FusedOperation::FUSED_OP_T& FusedOperation::type() const {
    return type_;
}
const Array& FusedOperation::array() const {
    return arr_;
}
const double& FusedOperation::scalar() const {
    return scalar_;
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

int FusedOperation::number_of_elements() const {
    if (type_ == FUSED_OP_ARRAY_T) {
        return arr_.number_of_elements();
    } else if (type_ == FUSED_OP_SCALAR_T) {
        return 1;
    } else {
        return hypercube_volume(shape());
    }
}

std::vector<int> FusedOperation::shape() const {
    auto res = bshape();
    for (auto& val : res) {
        if (val < 0) {
            val = std::abs(val);
        }
    }
    return res;
}

std::vector<int> FusedOperation::bshape() const {
    if (type_ == FUSED_OP_ARRAY_T) {
        return arr_.bshape();
    } else if (type_ == FUSED_OP_SCALAR_T ||
               type_ == FUSED_OP_ALLREDUCE_T ||
               type_ == FUSED_OP_ARGUMENT_ALLREDUCE_T) {
        return {};
    } else if (type_ == FUSED_OP_AXISREDUCE_LOW_DIM_T ||
               type_ == FUSED_OP_ARGUMENT_AXISREDUCE_LOW_DIM_T) {
        auto res = arguments_[0].bshape();
        res.erase(res.begin() + res.size() - 1, res.end());
        return res;
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
    if (type_ == FUSED_OP_ARRAY_T) {
        if (arr_.strides().empty()) {
            return 1;
        } else {
            return arr_.ndim();
        }
    } else if (type_ == FUSED_OP_ALLREDUCE_T ||
               type_ == FUSED_OP_ARGUMENT_ALLREDUCE_T) {
        return 1;
    } else if (type_ == FUSED_OP_AXISREDUCE_LOW_DIM_T ||
               type_ == FUSED_OP_ARGUMENT_AXISREDUCE_LOW_DIM_T) {
        return std::max(2, arguments_[0].computation_rank()) - 1;
    } else {
        int rank = FusedOperation::type_to_min_rank(type_);
        for (auto& arg : arguments_) {
            rank = std::max(rank, arg.computation_rank());
        }
        return rank;
    }
}

bool FusedOperation::dimension_is_contiguous_with_parent(const int& dim) const {
    if (type_ == FUSED_OP_ARRAY_T) {
        // TODO: make this check look at normalized strides
        // where possible (ensures that less code gets compiled)
        return arr_.strides().empty();
    } else if (type_ == FUSED_OP_ALLREDUCE_T ||
               type_ == FUSED_OP_ARGUMENT_ALLREDUCE_T) {
        return false;
    } else if (type_ == FUSED_OP_AXISREDUCE_LOW_DIM_T ||
               type_ == FUSED_OP_ARGUMENT_AXISREDUCE_LOW_DIM_T) {
        bool is_contig = true;
        for (auto& arg : arguments_) {
            is_contig = is_contig && arg.dimension_is_contiguous_with_parent(dim - 1);
        }
        return is_contig;
    } else {
        bool is_contig = true;
        for (auto& arg : arguments_) {
            is_contig = is_contig && arg.dimension_is_contiguous_with_parent(dim);
        }
        return is_contig;
    }
}

void FusedOperation::collapse_dimension_with_parent(const int& dim) {
    if (type_ == FUSED_OP_ARRAY_T) {
        std::vector<int> newshape = arr_.shape();
        newshape[dim - 1] = newshape[dim] * newshape[dim - 1];
        newshape.erase(newshape.begin() + dim);
        arr_ = arr_.copyless_reshape(newshape);
    } else if (type_ == FUSED_OP_ALLREDUCE_T ||
               type_ == FUSED_OP_ARGUMENT_ALLREDUCE_T) {
    } else if (type_ == FUSED_OP_AXISREDUCE_LOW_DIM_T ||
               type_ == FUSED_OP_ARGUMENT_AXISREDUCE_LOW_DIM_T) {
        for (auto& arg : arguments_) {
            arg.collapse_dimension_with_parent(dim - 1);
        }
    } else {
        for (auto& arg : arguments_) {
            arg.collapse_dimension_with_parent(dim);
        }
    }
}

void FusedOperation::transpose(const std::vector<int>& axes) {
    if (type_ == FUSED_OP_ARRAY_T) {
        arr_ = arr_.transpose(axes);
    } else if (type_ == FUSED_OP_ALLREDUCE_T ||
               type_ == FUSED_OP_ARGUMENT_ALLREDUCE_T) {
    } else if (type_ == FUSED_OP_AXISREDUCE_LOW_DIM_T ||
               type_ == FUSED_OP_ARGUMENT_AXISREDUCE_LOW_DIM_T) {
        auto extended_axes = axes;
        // ensure last axis does not move from current position:
        extended_axes.emplace_back(extended_axes.size());
        for (auto& arg : arguments_) {
            arg.transpose(extended_axes);
        }
    } else {
        for (auto& arg : arguments_) {
            arg.transpose(axes);
        }
    }
}


std::string FusedOperation::get_code_setup(memory::Device device, int rank, int& arg_idx, int& scalar_arg_idx) const {
    if (type_ == FUSED_OP_ARRAY_T) {
        auto res = build_views_constructor(dtype_to_cpp_name(dtype_), {arr_.strides().empty()}, rank, arg_idx);
        arg_idx += 1;
        return res;
    } else if (type_ == FUSED_OP_SCALAR_T) {
        auto res = build_scalar_constructor(dtype_to_cpp_name(dtype_), rank, scalar_arg_idx);
        scalar_arg_idx += 1;
        return res;
    } else if (type_ == FUSED_OP_ALLREDUCE_T ||
               type_ == FUSED_OP_ARGUMENT_ALLREDUCE_T) {
        // change in supported rank at this stage:
        utils::MS stream;
        for (auto& arg : arguments_) {
            stream << arg.get_code_setup(device, arg.computation_rank(), arg_idx, scalar_arg_idx);
        }
        return stream;
    } else if (type_ == FUSED_OP_AXISREDUCE_LOW_DIM_T ||
               type_ == FUSED_OP_ARGUMENT_AXISREDUCE_LOW_DIM_T) {
        // change in supported rank at this stage:
        utils::MS stream;
        for (auto& arg : arguments_) {
            stream << arg.get_code_setup(
                device,
                rank + 1,
                arg_idx,
                scalar_arg_idx
            );
        }
        return stream;
    } else {
        utils::MS stream;
        for (auto& arg : arguments_) {
            stream << arg.get_code_setup(device, rank, arg_idx, scalar_arg_idx);
        }
        return stream;
    }
}

std::string FusedOperation::get_code_setup(memory::Device device, int rank) const {
    int arg_idx = 0;
    int scalar_arg_idx = 0;
    return get_code_setup(device, rank, arg_idx, scalar_arg_idx);
}

std::string FusedOperation::get_call_nd(int rank) const {
    if (rank == 1) {
        return "(i)";
    } else {
        return "[query]";
    }
}

std::string FusedOperation::get_call_code_nd(int& arg_idx, int& scalar_arg_idx) const {
    if (type_ == FUSED_OP_ARRAY_T) {
        auto res = utils::make_message("arg_", arg_idx, "_view");
        arg_idx += 1;
        return res;
    } else if (type_ == FUSED_OP_SCALAR_T) {
        auto res = utils::make_message("scalar_", scalar_arg_idx);
        scalar_arg_idx += 1;
        return res;
    } else if (type_ == FUSED_OP_KERNEL_T ||
               type_ == FUSED_OP_ELEMENTWISE_T ||
               type_ == FUSED_OP_ALLREDUCE_T ||
               type_ == FUSED_OP_ARGUMENT_ALLREDUCE_T ||
               type_ == FUSED_OP_AXISREDUCE_LOW_DIM_T ||
               type_ == FUSED_OP_ARGUMENT_AXISREDUCE_LOW_DIM_T) {
        utils::MS stream;
        if (type_ == FUSED_OP_ELEMENTWISE_T) {
            stream << "element_wise_kernel<" << functor_name_ << ", " << dtype_to_cpp_name(dtype_) << ">(";
        } else if (type_ == FUSED_OP_KERNEL_T) {
            stream << functor_name_ << "(";
        } else if (type_ == FUSED_OP_ALLREDUCE_T) {
            ASSERT2(!arguments_.empty(), "all_reduce operation must have at least one argument.");
            int all_reduce_comp_rank = arguments_[0].computation_rank();
            stream << "all_reduce_kernel_" << all_reduce_comp_rank << "d<" << functor_name_
                   << ", " << dtype_to_cpp_name(dtype_) << ">(";
        } else if (type_ == FUSED_OP_ARGUMENT_ALLREDUCE_T) {
            ASSERT2(!arguments_.empty(), "argument_all_reduce operation must have at least one argument.");
            int all_reduce_comp_rank = arguments_[0].computation_rank();
            stream << "argument_all_reduce_kernel_" << all_reduce_comp_rank << "d<" << functor_name_
                   << ", " << dtype_to_cpp_name(dtype_) << ">(";
        } else if (type_ == FUSED_OP_AXISREDUCE_LOW_DIM_T) {
            ASSERT2(!arguments_.empty(), "axis_reduce operation must have at least one argument.");
            int axis_reduce_comp_rank = std::max(2, arguments_[0].computation_rank());
            stream << "axis_reduce_kernel_" << axis_reduce_comp_rank << "d<" << functor_name_
                   << ", " << dtype_to_cpp_name(dtype_) << ">(";
        } else if (type_ == FUSED_OP_ARGUMENT_AXISREDUCE_LOW_DIM_T) {
            ASSERT2(!arguments_.empty(), "argument_axis_reduce operation must have at least one argument.");
            int axis_reduce_comp_rank = std::max(2, arguments_[0].computation_rank());
            stream << "argument_axis_reduce_kernel_" << axis_reduce_comp_rank << "d<" << functor_name_
                   << ", " << dtype_to_cpp_name(dtype_) << ">(";
        }
        int args_called = 0;
        for (auto& arg : arguments_) {
            stream << arg.get_call_code_nd(arg_idx, scalar_arg_idx);
            args_called += 1;
            if (args_called != arguments_.size()) {
                stream << ", ";
            }
        }
        stream << ")";
        return stream;
    } else {
        ASSERT2(false, utils::make_message(
            "Unknown operation type (", type_, "). Use 0, 1, 2 3, 4, 5, 6, or 7."
        ));
    }
}

std::string FusedOperation::get_call_code_nd() const {
    int arg_idx = 0;
    int scalar_arg_idx = 0;
    return get_call_code_nd(arg_idx, scalar_arg_idx);
}

std::string FusedOperation::get_assign_code_nd(const OPERATOR_T& operator_t, const std::string& call_nd) const {
    return utils::make_message(
        "dst_view", call_nd, " ",
        operator_to_name(operator_t),
        " ", get_call_code_nd(), call_nd, ";\n"
    );
}

#define DALI_KEEP_TRACK_OF_GENERATED_CODE(GROUPING)\
    private:\
        std::unordered_set<int> GROUPING##s_;\
    public:\
        bool is_##GROUPING##_generated(int size) const {\
            return GROUPING##s_.find(size) != GROUPING##s_.end();\
        }\
        void mark_##GROUPING##_generated(int size) {\
            GROUPING##s_.insert(size);\
        }\

class GeneratedCodeTracker {
    DALI_KEEP_TRACK_OF_GENERATED_CODE(elementwise_kernel);
    DALI_KEEP_TRACK_OF_GENERATED_CODE(all_reduce_kernel);
    DALI_KEEP_TRACK_OF_GENERATED_CODE(argument_all_reduce_kernel);
    DALI_KEEP_TRACK_OF_GENERATED_CODE(axis_reduce_kernel);
    DALI_KEEP_TRACK_OF_GENERATED_CODE(argument_axis_reduce_kernel);
};

void fused_operation_get_extra_code(const FusedOperation& fop,
                                    GeneratedCodeTracker* tracker,
                                    std::string* extra_code_ptr) {
    if (!fop.extra_code().empty()) {
        *extra_code_ptr = (*extra_code_ptr) + fop.extra_code();
    }
    if (fop.type() == FusedOperation::FUSED_OP_ELEMENTWISE_T &&
        !tracker->is_elementwise_kernel_generated(fop.arguments().size())) {
        (*extra_code_ptr) = (
            (*extra_code_ptr) +
            create_elementwise_kernel_caller(fop.arguments().size())
        );
        tracker->mark_elementwise_kernel_generated(fop.arguments().size());
    }
    if (fop.type() == FusedOperation::FUSED_OP_ALLREDUCE_T &&
        !tracker->is_all_reduce_kernel_generated(fop.arguments()[0].computation_rank())) {
        (*extra_code_ptr) = (
            (*extra_code_ptr) +
            create_all_reduce_kernel_caller(fop.arguments()[0].computation_rank(), fop.computation_rank())
        );
        tracker->mark_all_reduce_kernel_generated(fop.arguments()[0].computation_rank());
    }
    if (fop.type() == FusedOperation::FUSED_OP_ARGUMENT_ALLREDUCE_T &&
        !tracker->is_argument_all_reduce_kernel_generated(fop.arguments()[0].computation_rank())) {
        (*extra_code_ptr) = (
            (*extra_code_ptr) +
            create_argument_all_reduce_kernel_caller(fop.arguments()[0].computation_rank(), fop.computation_rank())
        );
        tracker->mark_argument_all_reduce_kernel_generated(fop.arguments()[0].computation_rank());
    }
    if (fop.type() == FusedOperation::FUSED_OP_AXISREDUCE_LOW_DIM_T &&
        !tracker->is_axis_reduce_kernel_generated(std::max(2, fop.arguments()[0].computation_rank()))) {
        (*extra_code_ptr) = (
            (*extra_code_ptr) +
            create_axis_reduce_kernel_caller(std::max(2, fop.arguments()[0].computation_rank()))
        );
        tracker->mark_axis_reduce_kernel_generated(std::max(2, fop.arguments()[0].computation_rank()));
    }
    if (fop.type() == FusedOperation::FUSED_OP_ARGUMENT_AXISREDUCE_LOW_DIM_T &&
        !tracker->is_argument_axis_reduce_kernel_generated(std::max(2, fop.arguments()[0].computation_rank()))) {
        (*extra_code_ptr) = (
            (*extra_code_ptr) +
            create_argument_axis_reduce_kernel_caller(std::max(2, fop.arguments()[0].computation_rank()))
        );
        tracker->mark_argument_axis_reduce_kernel_generated(std::max(2, fop.arguments()[0].computation_rank()));
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


std::string FusedOperation::get_code_template(const OPERATOR_T& operator_t, bool dst_contiguous, DType output_dtype, memory::Device device, int rank) const {
    std::string code = utils::make_message(
        get_extra_code(),
        "void run(Array& dst, const std::vector<Array>& arguments, const std::vector<double>& scalar_arguments) {\n"
    );
    code += get_code_setup(device, rank);
    code += build_view_constructor(
        dtype_to_cpp_name(output_dtype), dst_contiguous, rank, "dst"
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
            "        ", get_assign_code_nd(operator_t, call_nd),
            "    }\n"
        );
    } else {
        for_loop = construct_for_loop(
            rank,
            get_assign_code_nd(operator_t, call_nd),
            "dst_view",
            4
        );
    }
    code += for_loop;
    code += "}\n";
    return code;
}

std::function<void(Array&, const std::vector<Array>&, const std::vector<double>&)> FusedOperation::compile(
        const OPERATOR_T& operator_t, bool dst_contiguous, DType output_dtype, memory::Device device) const {
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
            output_dtype,
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
            output_dtype,
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

        auto arrays = fop.get_arrays(output_bshape, fop.computation_rank());
        auto scalars = fop.get_scalars();
        fptr(out, arrays, scalars);
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
    int a_ndim = a.ndim();
    int b_ndim = b.ndim();
    return a_ndim == 0 || b_ndim == 0 || a_ndim == b_ndim;
}

DType FusedOperation::type_promotion(const FusedOperation& a, const FusedOperation& b) {
    // TODO(jonathan,szymon) speed up this function
    bool a_scalar = a.is_scalar();
    bool b_scalar = b.is_scalar();
    if (a_scalar && b_scalar || (!a_scalar && !b_scalar)) {
        if (a.dtype_ == DTYPE_DOUBLE || b.dtype_ == DTYPE_DOUBLE) {
            return DTYPE_DOUBLE;
        } else if (a.dtype_ == DTYPE_FLOAT || b.dtype_ == DTYPE_FLOAT) {
            return DTYPE_FLOAT;
        } else {
            return DTYPE_INT32;
        }
    } else if (a_scalar) {
        return b.dtype_;
    } else {
        return a.dtype_;
    }
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
    FusedOperation all_reduce(const FusedOperation& a,
                              const std::string& reducer_name,
                              DType return_type) {
        return FusedOperation(
            FusedOperation::FUSED_OP_ALLREDUCE_T,
            reducer_name,
            {a},
            return_type
        );
    }

    FusedOperation all_reduce(const FusedOperation& a,
                              const std::string& reducer_name) {
        return all_reduce(a, reducer_name, a.dtype());
    }

    FusedOperation axis_reduce(const FusedOperation& a,
                               const std::string& reducer_name,
                               const std::vector<int>& axes,
                               DType return_type) {
        if (axes.size() == 0) return a;
        int ndim = a.ndim();
        if (ndim == 0) return a;
        std::vector<int> normalized_axes(axes);
        for (auto& axis : normalized_axes) {
            if (axis < 0) {
                if (ndim == 0) {
                    axis = axis + 1;
                } else {
                    axis = axis + ndim;
                }
            }
            ASSERT2(axis >= 0 && (axis < ndim || ndim == 0 && axis == ndim),
                utils::make_message(
                    "Reduction axis must strictly positive and less than the "
                    "number of dimensions of the input (got axis=", axes[0], ","
                    " ndim=", ndim, ")."
                )
            );
        }
        // now look to see what kind of a reduction this is:
        std::vector<bool> reduced_dims(ndim, false);
        std::sort(normalized_axes.begin(), normalized_axes.end());
        for (auto& axis : normalized_axes) {
            ASSERT2(!reduced_dims[axis], utils::make_message("axis_reduce "
                "received duplicate axes to operate on (axis=", axis,
                " axes=", axes, ")."
            ));
            reduced_dims[axis] = true;
        }
        // all axes are present:
        if (normalized_axes.size() == ndim) {
            return all_reduce(a, reducer_name, return_type);
        }
        int num_low_dims = 0;
        for (int i = reduced_dims.size() - 1; i >= 0; --i) {
            if (reduced_dims[i]) {
                ++num_low_dims;
            } else {
                break;
            }
        }
        bool all_reductions_are_low_dim = num_low_dims == normalized_axes.size();
        auto res = a;

        if (!all_reductions_are_low_dim) {
            std::vector<int> new_axes_order;
            for (int i = 0; i < reduced_dims.size(); ++i) {
                if (!reduced_dims[i]) {
                    new_axes_order.emplace_back(i);
                }
            }
            for (int i = 0; i < reduced_dims.size(); ++i) {
                if (reduced_dims[i]) {
                    new_axes_order.emplace_back(i);
                }
            }
            res.transpose(new_axes_order);
        }
        int num_low_axes_to_reduce = normalized_axes.size();
        if (num_low_axes_to_reduce > 0) {
            int axes_used_up = 0;
            int collapsed_ndim = ndim - 1;
            for (int axes_used_up = 0; axes_used_up < num_low_axes_to_reduce; ++axes_used_up) {
                if (num_low_axes_to_reduce - axes_used_up == 1) {
                    res = FusedOperation(
                        FusedOperation::FUSED_OP_AXISREDUCE_LOW_DIM_T,
                        reducer_name,
                        {res},
                        return_type
                    );
                } else {
                    if (res.dimension_is_contiguous_with_parent(collapsed_ndim)) {
                        res.collapse_dimension_with_parent(collapsed_ndim);
                    } else {
                        res = FusedOperation(
                            FusedOperation::FUSED_OP_AXISREDUCE_LOW_DIM_T,
                            reducer_name,
                            {res},
                            return_type
                        );
                    }
                }
                --collapsed_ndim;
            }
        }
        return res;
    }

    FusedOperation axis_reduce(const FusedOperation& a,
                               const std::string& reducer_name,
                               const std::vector<int>& axes) {
        return axis_reduce(a, reducer_name, axes, a.dtype());
    }

    FusedOperation argument_all_reduce(const FusedOperation& a,
                                       const std::string& reducer_name) {
        return FusedOperation(
            FusedOperation::FUSED_OP_ARGUMENT_ALLREDUCE_T,
            reducer_name,
            {a},
            DTYPE_INT32
        );
    }

    FusedOperation argument_axis_reduce(const FusedOperation& a,
                                        const std::string& reducer_name,
                                        const int& axis) {
        int ndim = a.ndim();
        if (ndim == 0) return FusedOperation(0);
        int normalized_axis = axis;
        if (normalized_axis < 0) normalized_axis = normalized_axis + a.ndim();
        ASSERT2(normalized_axis >= 0 && (normalized_axis < ndim || ndim == 0 && normalized_axis == ndim),
            utils::make_message(
                "Reduction axis must strictly positive and less than the "
                "number of dimensions of the input (got axis=", normalized_axis, ","
                " ndim=", ndim, ")."
            )
        );
        if (ndim == 1) return argument_all_reduce(a, reducer_name);

        auto res = a;
        if (normalized_axis != ndim - 1) {
            std::vector<int> axes;
            for (int i = 0; i < ndim; i++) {
                axes.emplace_back(i);
            }
            axes[axes.size() - 1] = normalized_axis;
            axes[normalized_axis] = axes.size() - 1;
            res.transpose(axes);
        }
        return FusedOperation(
            FusedOperation::FUSED_OP_ARGUMENT_AXISREDUCE_LOW_DIM_T,
            reducer_name,
            {res},
            DTYPE_INT32
        );
    }

    FusedOperation elementwise(
        const FusedOperation& a,
        const std::string& functor_name,
        DType return_type) {

        return FusedOperation(
            FusedOperation::FUSED_OP_ELEMENTWISE_T,
            functor_name,
            {a},
            return_type
        );
    }

    FusedOperation elementwise(
        const FusedOperation& a,
        const std::string& functor_name) {
        return elementwise(
            a, functor_name, a.dtype()
        );
    }

    FusedOperation elementwise(
        const FusedOperation& a,
        const FusedOperation& b,
        const std::string& functor_name) {

        // perform type promotion:
        if (a.dtype() != b.dtype()) {
            auto new_type = FusedOperation::type_promotion(a, b);
            if (a.dtype() == new_type) {
                // b's dtype is being promoted
                return elementwise(
                    a,
                    astype(b, new_type),
                    functor_name
                );
            } else {
                // a's dtype is being promoted
                return elementwise(
                    astype(a, new_type),
                    b,
                    functor_name
                );
            }
        } else {
            ASSERT2(FusedOperation::ndim_compatible(a, b), "ranks don't match");
            auto output_bshape = get_function_bshape(a, b);
            return FusedOperation(
                FusedOperation::FUSED_OP_ELEMENTWISE_T,
                functor_name,
                {a, b},
                a.dtype()
            );
        }
    }

    FusedOperation binary_kernel_function(
        const FusedOperation& a,
        const FusedOperation& b,
        const std::string& function_name,
        const std::string& kernel_code
    ) {
        // perform type promotion:
        if (a.dtype() != b.dtype()) {
            auto new_type = FusedOperation::type_promotion(a, b);
            if (a.dtype() == new_type) {
                // b's dtype is being promoted
                return binary_kernel_function(
                    a,
                    astype(b, new_type),
                    function_name,
                    kernel_code
                );
            } else {
                // a's dtype is being promoted
                return binary_kernel_function(
                    astype(a, new_type),
                    b,
                    function_name,
                    kernel_code
                );
            }
        } else {
            ASSERT2(FusedOperation::ndim_compatible(a, b), "ranks don't match");
            auto output_bshape = get_function_bshape(a, b);

            return FusedOperation(
                FusedOperation::FUSED_OP_KERNEL_T,
                function_name,
                kernel_code,
                {a, b},
                a.dtype()
            );
        }
    }

    FusedOperation astype(const FusedOperation& x, DType type) {
        return elementwise(x, "functor::cast", type);
    }
}

