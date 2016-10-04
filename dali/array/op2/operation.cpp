#include "operation.h"

#include <unordered_set>

#include "dali/array/function2/compiler.h"
#include "dali/array/op2/all_reduce_kernel_utils.h"
#include "dali/array/op2/elementwise_kernel_utils.h"
#include "dali/array/op2/rtc_utils.h"
#include "dali/utils/make_message.h"

using utils::Hasher;

///////////////////////////////////////////////////////////////////////////////
//                         MISC UTILS                                        //
///////////////////////////////////////////////////////////////////////////////

DType type_promotion(const Operation& a, const Operation& b) {
    // TODO(jonathan,szymon) speed up this function
    bool a_scalar = a.is_scalar();
    bool b_scalar = b.is_scalar();

    if (a_scalar ^ b_scalar == 0) {
        // if they are both scalars or both arrays
        if (a.dtype() == DTYPE_DOUBLE || b.dtype() == DTYPE_DOUBLE) {
            return DTYPE_DOUBLE;
        } else if (a.dtype() == DTYPE_FLOAT || b.dtype() == DTYPE_FLOAT) {
            return DTYPE_FLOAT;
        } else {
            return DTYPE_INT32;
        }
    } else if (a_scalar) {
        // if a is scalar and b is array.
        return b.dtype();
    } else {
        // if a is array and b is scalar.
        return a.dtype();
    }
}

bool ndim_compatible(const Operation& a, const Operation& b) {
    int a_ndim = a.ndim();
    int b_ndim = b.ndim();
    return a_ndim == 0 || b_ndim == 0 || a_ndim == b_ndim;
}


std::string indexing_code_nd(int rank) {
    if (rank == 1) {
        return "(i)";
    } else {
        return "[query]";
    }
}




///////////////////////////////////////////////////////////////////////////////
//                         OPERATION STATE                                   //
///////////////////////////////////////////////////////////////////////////////

std::vector<int> OperationState::shape() const {
    std::vector<int> res = bshape();
    std::transform(res.begin(), res.end(), res.begin(),
        [](const int& x) {return std::abs(x);}
    );
    return res;
}

int OperationState::number_of_elements() const {
    return hypercube_volume(shape());
}


std::string OperationState::prefix_code(const node_to_info_t& node_to_info) const {
    return "";
}


std::vector<operation_state_ptr> OperationState::arguments() const {
    return {};
}

bool OperationState::is_dim_collapsible_with_dim_minus_one(const int& dim) const {
    return false;
}

operation_state_ptr OperationState::collapse_dim_with_dim_minus_one(const int& dim) const {
    return shared_from_this();
}

operation_state_ptr OperationState::transpose(const std::vector<int>& permutation) const {
    ASSERT2(false, "Transpose not implemented for this Operation.");
    return shared_from_this();
}

OperationState::OperationState(int min_computation_rank) : min_computation_rank_(min_computation_rank) {}


std::string OperationState::get_code_template(const OPERATOR_T& operator_t,
                                              bool dst_contiguous,
                                              DType output_dtype,
                                              memory::Device device,
                                              int desired_computation_rank,
                                              const std::vector<const ArrayOperationState*>& arrays,
                                              const std::vector<const ScalarOperationState*>& scalars,
                                              const node_to_info_t& node_to_info) const {
    std::unordered_set<hash_t> prefix_code_visited;
    std::stringstream result;

    this->for_all_suboperations([&](const OperationState* node) {
        auto pc      = node->prefix_code(node_to_info);
        auto pc_hash = utils::get_hash(pc);
        if (prefix_code_visited.find(pc_hash) == prefix_code_visited.end()) {
            result << pc;
            prefix_code_visited.insert(pc_hash);
        }
    });

    result << "void run(Array& dst, const std::vector<Array>& array_arguments, const std::vector<double>& scalar_arguments) {\n";

    // DECLARE SYMBOLS

    symbol_table_t symbol_table;
    for (int i = 0; i < arrays.size(); ++i) {
        auto name = utils::make_message("array_", i, "_view");

        symbol_table[(const OperationState*)arrays[i]] = name;
        result << build_array_definition(
            dtype_to_cpp_name(arrays[i]->dtype()),
            name,
            arrays[i]->contiguous(),
            node_to_info.at(arrays[i]).computation_rank,
            utils::make_message("array_arguments[", i, "]")
        );
    }

    for (int i = 0; i < scalars.size(); ++i) {
        auto name = utils::make_message("scalar_", i, "_view");

        symbol_table[(const OperationState*)scalars[i]] = name;

        result << build_scalar_definition(
            dtype_to_cpp_name(scalars[i]->dtype()),
            name,
            node_to_info.at(scalars[i]).computation_rank,
            utils::make_message("scalar_arguments[", i, "]")
        );
    }

    result << build_array_definition(
        dtype_to_cpp_name(output_dtype), "dst_view", dst_contiguous, desired_computation_rank, "dst"
    );

    // now we declare output.
    std::string for_loop;
    std::string indexing_nd = indexing_code_nd(desired_computation_rank);
    if (desired_computation_rank == 1) {
        for_loop = utils::make_message(
            "    int num_el = dst.number_of_elements();\n",
            "    #pragma clang loop vectorize(enable)\n",
            "    #pragma clang loop interleave(enable)\n",
            "    for (int i = 0; i < num_el; ++i) {\n",
            "        ", get_assign_code_nd(operator_t, indexing_nd, symbol_table, node_to_info),
            "    }\n"
        );
    } else {
        for_loop = construct_for_loop(
            desired_computation_rank,
            get_assign_code_nd(operator_t, indexing_nd, symbol_table, node_to_info),
            "dst_view",
            4
        );
    }
    result << for_loop;
    result << "}\n";
    return result.str();
}

std::string OperationState::get_assign_code_nd(const OPERATOR_T& operator_t,
                                                       const std::string& call_nd,
                                                       const symbol_table_t& symbol_table,
                                                       const node_to_info_t& node_to_info) const {
    return utils::make_message(
        "dst_view", call_nd, " ",
        operator_to_name(operator_t),
        " ", get_call_code_nd(symbol_table, node_to_info), call_nd, ";\n"
    );
}



std::function<void(Array&, const std::vector<Array>&, const std::vector<double>&)> OperationState::compile(
        const OPERATOR_T& operator_t,
        bool dst_contiguous,
        DType output_dtype,
        int desired_computation_rank,
        memory::Device device,
        const std::vector<const ArrayOperationState*>& arrays,
        const std::vector<const ScalarOperationState*>& scalars,
        const node_to_info_t& node_to_info) const {
    // compute a quasi-unique hash for the fused operation
    hash_t hash = Hasher().add((int)output_dtype)
                          .add((int)operator_t)
                          .add(device.is_cpu())
                          .add(node_to_info.at(this).hash)
                          .add(dst_contiguous)
                          .add(desired_computation_rank)
                          .value();
    // check if the operation needs to be runtime compiled
    if (!array_op_compiler.load(hash)) {
        auto code_template = get_code_template(
            operator_t,
            dst_contiguous,
            output_dtype,
            device,
            desired_computation_rank,
            arrays,
            scalars,
            node_to_info
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


OperationState::operator Assignable<Array> () const {
    auto this_ptr = shared_from_this();
    return Assignable<Array>([this_ptr](Array& out, const OPERATOR_T& operator_t) mutable {
        auto& self = *this_ptr;

        auto output_dtype  = self.dtype();
        auto output_device = memory::Device::cpu();
        auto output_bshape = self.bshape();


        initialize_output_array(
            out,
            output_dtype,
            output_device,
            &output_bshape
        );

        bool dst_contiguous = out.strides().empty();

        // get the lowest dimension that suffices to compute this problem:
        int desired_computation_rank = self.min_computation_rank_;
        // given the lowest rank for which the operation can be executed
        // now check if the destination demands a higher rank (due to striding)
        if (!dst_contiguous) {
            desired_computation_rank = std::max(desired_computation_rank, self.ndim());
        }

        std::vector<const ArrayOperationState*> array_ops;
        std::vector<const ScalarOperationState*> scalar_ops;
        node_to_info_t node_to_info;

        self.compute_node_compilation_info(desired_computation_rank,
                                           output_bshape,
                                           &array_ops,
                                           &scalar_ops,
                                           &node_to_info);


        auto compiled_self = self.compile(operator_t,
                                          dst_contiguous,
                                          output_dtype,
                                          desired_computation_rank,
                                          output_device,
                                          array_ops,
                                          scalar_ops,
                                          node_to_info);
        std::vector<Array> arrays;
        std::transform(array_ops.begin(),
                       array_ops.end(),
                       std::back_inserter(arrays),
                       [&node_to_info](const ArrayOperationState* op) {
                           const auto& rank  = node_to_info.at(op).computation_rank;
                           const auto& shape = node_to_info.at(op).computation_shape;
                           if (rank == op->ndim()) {
                               return op->array_.reshape_broadcasted(shape);
                           } else if (rank == 1) {
                               return op->array_.reshape_broadcasted(shape).copyless_ravel();
                           } else {
                               return op->array_.reshape_broadcasted(shape).copyless_right_fit_ndim(rank);
                           }
                       });
        auto out_reshaped = out;
        if (desired_computation_rank != out_reshaped.ndim()) {
            if (desired_computation_rank == 1) {
                out_reshaped = out.copyless_ravel();
            } else {
                out_reshaped = out.copyless_right_fit_ndim(desired_computation_rank);
            }
        }

        std::vector<double> scalars;
        std::transform(scalar_ops.begin(),
                       scalar_ops.end(),
                       std::back_inserter(scalars),
                       [&](const ScalarOperationState* op) {
                           return op->value_;
                       });

        compiled_self(out_reshaped, arrays, scalars);
    });
}

void OperationState::for_all_suboperations(std::function<void(const OperationState*)> callback) const {
    callback(this);
    for (auto& child: arguments()) {
        child->for_all_suboperations(callback);
    }
}




///////////////////////////////////////////////////////////////////////////////
//                   ARRAY OPERATION STATE                                   //
///////////////////////////////////////////////////////////////////////////////

const hash_t ArrayOperationState::optype_hash = std::hash<std::string>()("ArrayOperationState");

ArrayOperationState::ArrayOperationState(Array array) :
        OperationState(array.strides().empty() ? 1 : array.ndim()),
        array_(array) {
}

DType ArrayOperationState::dtype() const {
    return array_.dtype();
}

std::vector<int> ArrayOperationState::bshape() const {
    return array_.bshape();
}

int ArrayOperationState::ndim() const {
    return array_.ndim();
}

bool ArrayOperationState::contiguous() const {
    return array_.strides().empty();
}


std::vector<int> ArrayOperationState::shape() const {
    return array_.shape();
}

int ArrayOperationState::number_of_elements() const {
    return array_.number_of_elements();
}

void ArrayOperationState::compute_node_compilation_info(int desired_computation_rank,
                                                                const std::vector<int>& desired_computation_shape,
                                                                std::vector<const ArrayOperationState*>* arrays,
                                                                std::vector<const ScalarOperationState*>* scalars,
                                                                node_to_info_t* node_to_info) const {
    arrays->emplace_back(this);
    (*node_to_info)[this].computation_rank  = desired_computation_rank;
    (*node_to_info)[this].computation_shape = desired_computation_shape;
    (*node_to_info)[this].hash = Hasher().add(optype_hash)
                                          .add(desired_computation_rank)
                                          .add(contiguous()).value();
}

bool ArrayOperationState::is_dim_collapsible_with_dim_minus_one(const int& dim) const {
    // TODO(jonathan): have fun and
    // make this check look at normalized strides
    // where possible (ensures that less code gets compiled)
    // once this is implemented, reshape needs to be updated
    // to leverage this property.
    return contiguous();
}

operation_state_ptr ArrayOperationState::collapse_dim_with_dim_minus_one(const int& dim) const {
    std::vector<int> newshape = array_.shape();
    newshape[dim - 1] = newshape[dim] * newshape[dim - 1];
    newshape.erase(newshape.begin() + dim);
    return std::make_shared<ArrayOperationState>(array_.copyless_reshape(newshape));
}

operation_state_ptr ArrayOperationState::transpose(const std::vector<int>& permutation) const {
    return std::make_shared<ArrayOperationState>(array_.transpose(permutation));
}

std::string ArrayOperationState::get_call_code_nd(const symbol_table_t& symbol_table, const node_to_info_t& node_to_info) const {
    return symbol_table.at(this);
}


///////////////////////////////////////////////////////////////////////////////
//                   SCALAR OPERATION STATE                                  //
///////////////////////////////////////////////////////////////////////////////

const hash_t ScalarOperationState::optype_hash = std::hash<std::string>()("ScalarOperationState");

ScalarOperationState::ScalarOperationState(double value) : OperationState(1), value_(value) {}

DType ScalarOperationState::dtype() const {
    return DTYPE_DOUBLE;
}

std::vector<int> ScalarOperationState::bshape() const {
    return {};
}

int ScalarOperationState::ndim() const {
    return 0;
}

std::vector<int> ScalarOperationState::shape() const {
    return {};
}

int ScalarOperationState::number_of_elements() const {
    return 1;
}


void ScalarOperationState::compute_node_compilation_info(int desired_computation_rank,
                                           const std::vector<int>& desired_computation_shape,
                                           std::vector<const ArrayOperationState*>* arrays,
                                           std::vector<const ScalarOperationState*>* scalars,
                                           node_to_info_t* node_to_info) const {
    scalars->emplace_back(this);
    (*node_to_info)[this].computation_rank = desired_computation_rank;
    (*node_to_info)[this].hash = Hasher().add(optype_hash).add(desired_computation_rank).value();
}

bool ScalarOperationState::is_dim_collapsible_with_dim_minus_one(const int& dim) const {
    return true;
}

operation_state_ptr ScalarOperationState::transpose(const std::vector<int>& permutation) const {
    return shared_from_this();
}

std::string ScalarOperationState::get_call_code_nd(const symbol_table_t& symbol_table, const node_to_info_t& node_to_info) const {
    return symbol_table.at(this);
}

///////////////////////////////////////////////////////////////////////////////
//                   ELEMENTWISE OPERATION STATE                             //
///////////////////////////////////////////////////////////////////////////////

const hash_t ElementwiseOperationState::optype_hash = std::hash<std::string>()("ElementwiseOperationState");

int ElementwiseOperationState::compute_min_computation_rank(
        const operation_state_ptrs& arguments) {
    return std::accumulate(arguments.begin(),
                           arguments.end(),
                           0,
                           [](int so_far, operation_state_ptr op) {
                               return std::max(so_far, op->min_computation_rank_);
                           });
}

ElementwiseOperationState::ElementwiseOperationState(
    const std::string& functor_name,
    const operation_state_ptrs& arguments) :
        OperationState(compute_min_computation_rank(arguments)),
        functor_name_(functor_name),
        arguments_(arguments) {
}


DType ElementwiseOperationState::dtype() const {
    return arguments_[0]->dtype();
}

std::vector<int> ElementwiseOperationState::bshape() const {
    std::vector<std::vector<int>> arg_bshapes;
    for (auto& arg: arguments_) {
        arg_bshapes.emplace_back(arg->bshape());
    }
    return get_common_bshape(arg_bshapes);
}

int ElementwiseOperationState::ndim() const {
    return bshape().size();
}

std::vector<operation_state_ptr> ElementwiseOperationState::arguments() const { return arguments_; }

void ElementwiseOperationState::compute_node_compilation_info(
        int desired_computation_rank,
        const std::vector<int>& desired_computation_shape,
        std::vector<const ArrayOperationState*>* arrays,
        std::vector<const ScalarOperationState*>* scalars,
        node_to_info_t* node_to_info) const {
    (*node_to_info)[this].computation_rank = desired_computation_rank;
    for (auto& arg: arguments_) {
        arg->compute_node_compilation_info(desired_computation_rank, desired_computation_shape, arrays, scalars, node_to_info);
    }
    Hasher hasher;
    hasher.add(optype_hash).add(desired_computation_rank).add(functor_name_);
    for (auto& arg: arguments_) {
        hasher.add(node_to_info->at(arg.get()).hash);
    }
    (*node_to_info)[this].hash = hasher.value();
}

bool ElementwiseOperationState::is_dim_collapsible_with_dim_minus_one(const int& dim) const {
    bool is_contig = true;
    for (auto& arg : arguments_) {
        is_contig = is_contig && arg->is_dim_collapsible_with_dim_minus_one(dim);
    }
    return is_contig;
}

operation_state_ptr ElementwiseOperationState::collapse_dim_with_dim_minus_one(const int& dim) const {
    operation_state_ptrs new_arguments;

    for (auto& arg : arguments_) {
        new_arguments.emplace_back(arg->collapse_dim_with_dim_minus_one(dim));
    }

    return std::make_shared<ElementwiseOperationState>(functor_name_, new_arguments);
}

operation_state_ptr ElementwiseOperationState::transpose(const std::vector<int>& permutation) const {
    operation_state_ptrs new_arguments;

    for (auto& arg : arguments_) {
        new_arguments.emplace_back(arg->transpose(permutation));
    }

    return std::make_shared<ElementwiseOperationState>(functor_name_, new_arguments);
}

std::string ElementwiseOperationState::get_call_code_nd(
        const symbol_table_t& symbol_table,
        const node_to_info_t& node_to_info) const {
    std::stringstream stream;
    stream << "element_wise_kernel<" << functor_name_ << ", " << dtype_to_cpp_name(dtype()) << ">(";

    for (int i = 0; i < arguments_.size(); ++i) {
        stream << arguments_[i]->get_call_code_nd(symbol_table, node_to_info)
               << (i + 1 == arguments_.size() ? "" : ", ");
    }
    stream << ")";
    return stream.str();
}

std::string ElementwiseOperationState::prefix_code(
        const node_to_info_t& node_to_info) const {
    return create_elementwise_kernel_caller(arguments_.size());
}


///////////////////////////////////////////////////////////////////////////////
//                       CAST OPERATION STATE                                //
///////////////////////////////////////////////////////////////////////////////

const hash_t CastOperationState::optype_hash = std::hash<std::string>()("CastOperationState");


CastOperationState::CastOperationState(
        DType dtype,
        const operation_state_ptr argument) :
    ElementwiseOperationState("functor::cast", {argument}),
    dtype_(dtype) {
}

void CastOperationState::compute_node_compilation_info(
        int desired_computation_rank,
        const std::vector<int>& desired_computation_shape,
        std::vector<const ArrayOperationState*>* arrays,
        std::vector<const ScalarOperationState*>* scalars,
        node_to_info_t* node_to_info) const {
    (*node_to_info)[this].computation_rank = desired_computation_rank;
    arguments_[0]->compute_node_compilation_info(desired_computation_rank, desired_computation_shape, arrays, scalars, node_to_info);

    (*node_to_info)[this].hash = Hasher().add(optype_hash)
                                      .add(desired_computation_rank)
                                      .add(functor_name_)
                                      .add(node_to_info->at(arguments_[0].get()).hash)
                                      .add(dtype())
                                      .value();
}

DType CastOperationState::dtype() const {
    return dtype_;
}

///////////////////////////////////////////////////////////////////////////////
//                       REDUCER OPERATION STATE                             //
///////////////////////////////////////////////////////////////////////////////

const hash_t AllReducerOperationState::optype_hash_cache_ =
        std::hash<std::string>()("AllReducerOperationState");

ReducerOperationState::ReducerOperationState(
        const std::string& functor_name,
        operation_state_ptr argument,
        int min_computation_rank) :
    OperationState(min_computation_rank),
    functor_name_(functor_name),
    argument_(argument) {

}

std::vector<operation_state_ptr> ReducerOperationState::arguments() const {
    return {argument_};
}

std::string ReducerOperationState::get_call_code_nd(
        const symbol_table_t& symbol_table,
        const node_to_info_t& node_to_info) const {
    int all_reduce_comp_rank = node_to_info.at(argument_.get()).computation_rank;
    return utils::make_message(
        kernel_name(), all_reduce_comp_rank,
        "d<", functor_name_, ", " , dtype_to_cpp_name(dtype()) , ">(",
        argument_->get_call_code_nd(symbol_table, node_to_info), ")");

}

///////////////////////////////////////////////////////////////////////////////
//                    ALL REDUCER OPERATION STATE                            //
///////////////////////////////////////////////////////////////////////////////


const hash_t AxisReducerOperationState::optype_hash_cache_ = std::hash<std::string>()("AxisReducerOperationState");

hash_t AllReducerOperationState::optype_hash() const {
    return optype_hash_cache_;
}

AllReducerOperationState::AllReducerOperationState(
        const std::string& functor_name,
        operation_state_ptr argument) :
    ReducerOperationState(functor_name, argument, 1) {
}

std::vector<int> AllReducerOperationState::bshape() const {
    return {};
}

DType AllReducerOperationState::dtype() const {
    return argument_->dtype();
}

int AllReducerOperationState::ndim() const {
    return 0;
}

void AllReducerOperationState::compute_node_compilation_info(
        int desired_computation_rank,
        const std::vector<int>& desired_computation_shape,
        std::vector<const ArrayOperationState*>* arrays,
        std::vector<const ScalarOperationState*>* scalars,
        node_to_info_t* node_to_info) const {
    (*node_to_info)[this].computation_rank = desired_computation_rank;
    argument_->compute_node_compilation_info(argument_->min_computation_rank_, argument_->shape(), arrays, scalars, node_to_info);
    (*node_to_info)[this].hash = Hasher().add(optype_hash())
                                          .add(desired_computation_rank)
                                          .add(functor_name_)
                                          .add(node_to_info->at(argument_.get()).hash)
                                          .value();
}


bool AllReducerOperationState::is_dim_collapsible_with_dim_minus_one(
        const int& dim) const {
    return true;
}

operation_state_ptr AllReducerOperationState::transpose(
        const std::vector<int>& permutation) const {
    return shared_from_this();
}

std::string AllReducerOperationState::prefix_code(
        const node_to_info_t& node_to_info) const {
    return create_all_reduce_kernel_caller(
        node_to_info.at(argument_.get()).computation_rank,
        node_to_info.at(this).computation_rank
    );
}

std::string AllReducerOperationState::kernel_name() const {
    return "all_reduce_kernel_";
}



///////////////////////////////////////////////////////////////////////////////
//                    AXIS REDUCER OPERATION STATE                           //
///////////////////////////////////////////////////////////////////////////////

const hash_t ArgumentAllReducerOperationState::optype_hash_cache_ = std::hash<std::string>()("ArgumentAllReducerOperationState");

hash_t AxisReducerOperationState::optype_hash() const {
    return optype_hash_cache_;
}


AxisReducerOperationState::AxisReducerOperationState(
        const std::string& functor_name,
        operation_state_ptr argument) :
    ReducerOperationState(functor_name, argument, std::max(argument->min_computation_rank_ - 1, 1)) {
}

std::vector<int> AxisReducerOperationState::bshape() const {
    auto result = argument_->bshape();
    result.pop_back();
    return result;
}

DType AxisReducerOperationState::dtype() const {
    return argument_->dtype();
}

int AxisReducerOperationState::ndim() const {
    return std::max(argument_->ndim() - 1, 0);
}

void AxisReducerOperationState::compute_node_compilation_info(
        int desired_computation_rank,
        const std::vector<int>& desired_computation_shape,
        std::vector<const ArrayOperationState*>* arrays,
        std::vector<const ScalarOperationState*>* scalars,
        node_to_info_t* node_to_info) const {
    (*node_to_info)[this].computation_rank = desired_computation_rank;

    auto desired_argument_shape = desired_computation_shape;
    desired_argument_shape.emplace_back(argument_->shape().back());


    argument_->compute_node_compilation_info(desired_computation_rank + 1, desired_argument_shape, arrays, scalars, node_to_info);

    (*node_to_info)[this].hash = Hasher().add(optype_hash())
                                      .add(desired_computation_rank)
                                      .add(functor_name_)
                                      .add(node_to_info->at(argument_.get()).hash)
                                      .value();
}

bool AxisReducerOperationState::is_dim_collapsible_with_dim_minus_one(
        const int& dim) const {
    return argument_->is_dim_collapsible_with_dim_minus_one(dim - 1);;
}

operation_state_ptr AxisReducerOperationState::collapse_dim_with_dim_minus_one(
        const int& dim) const {
    return std::make_shared<AxisReducerOperationState>(
        functor_name_,
        argument_->collapse_dim_with_dim_minus_one(dim - 1)
    );
}

operation_state_ptr AxisReducerOperationState::transpose(
        const std::vector<int>& permutation) const {
    auto new_permutation = permutation;
    // add last dim of tensor with rank (permutation.size() + 1)
    new_permutation.emplace_back(permutation.size());

    return std::make_shared<AxisReducerOperationState>(
        functor_name_,
        argument_->transpose(new_permutation)
    );
}

std::string AxisReducerOperationState::prefix_code(
        const node_to_info_t& node_to_info) const {
    return create_axis_reduce_kernel_caller(node_to_info.at(argument_.get()).computation_rank);
}

std::string AxisReducerOperationState::kernel_name() const {
    return "axis_reduce_kernel_";
}


///////////////////////////////////////////////////////////////////////////////
//                ARGUMENT ALL REDUCER OPERATION STATE                       //
///////////////////////////////////////////////////////////////////////////////



hash_t ArgumentAllReducerOperationState::optype_hash() const {
    return optype_hash_cache_;
}

DType ArgumentAllReducerOperationState::dtype() const {
    return DTYPE_INT32;
}

std::string ArgumentAllReducerOperationState::prefix_code(
        const node_to_info_t& node_to_info) const {
    return create_argument_all_reduce_kernel_caller(
        node_to_info.at(argument_.get()).computation_rank,
        node_to_info.at(this).computation_rank
    );
}

std::string ArgumentAllReducerOperationState::kernel_name() const {
    return "argument_all_reduce_kernel_";
}


///////////////////////////////////////////////////////////////////////////////
//         ARGUMENT AXIS REDUCER OPERATION STATE                             //
///////////////////////////////////////////////////////////////////////////////

const hash_t ArgumentAxisReducerOperationState::optype_hash_cache_ = std::hash<std::string>()("ArgumentAxisReducerOperationState");

hash_t ArgumentAxisReducerOperationState::optype_hash() const {
    return optype_hash_cache_;
}

DType ArgumentAxisReducerOperationState::dtype() const {
    return DTYPE_INT32;
}

std::string ArgumentAxisReducerOperationState::prefix_code(
        const node_to_info_t& node_to_info) const {
    return create_argument_axis_reduce_kernel_caller(node_to_info.at(argument_.get()).computation_rank);
}

operation_state_ptr ArgumentAxisReducerOperationState::collapse_dim_with_dim_minus_one(
        const int& dim) const {
    return std::make_shared<ArgumentAxisReducerOperationState>(
        functor_name_,
        argument_->collapse_dim_with_dim_minus_one(dim - 1)
    );
}

operation_state_ptr ArgumentAxisReducerOperationState::transpose(
        const std::vector<int>& permutation) const {
    auto new_permutation = permutation;
    // add last dim of tensor with rank (permutation.size() + 1)
    new_permutation.emplace_back(permutation.size());

    return std::make_shared<ArgumentAxisReducerOperationState>(
        functor_name_,
        argument_->transpose(new_permutation)
    );
}

std::string ArgumentAxisReducerOperationState::kernel_name() const {
    return "argument_axis_reduce_kernel_";
}


///////////////////////////////////////////////////////////////////////////////
//                         OPERATION                                         //
///////////////////////////////////////////////////////////////////////////////

Operation::Operation(const Array& arr): Operation(std::make_shared<ArrayOperationState>(arr)) {
}

Operation::Operation(const Assignable<Array>& arr): Operation(std::make_shared<ArrayOperationState>(Array(arr))) {
}

Operation::Operation(double scalar): Operation(std::make_shared<ScalarOperationState>(scalar)) {
}

Operation::Operation(int scalar): Operation(op2::astype((double)scalar, DTYPE_INT32)) {
}

Operation::Operation(operation_state_ptr state): state_(state) {
}

DType Operation::dtype() const {
    return state_->dtype();
}

int Operation::ndim() const {
    return state_->ndim();
}

std::vector<int> Operation::bshape() const {
    return state_->bshape();
}

bool Operation::is_scalar() const {
    return ndim() == 0;
}

int Operation::number_of_elements() const {
    return state_->number_of_elements();
}




bool Operation::is_dim_collapsible_with_dim_minus_one(const int& dim) const {
    return state_->is_dim_collapsible_with_dim_minus_one(dim);
}

Operation Operation::collapse_dim_with_dim_minus_one(const int& dim) const {
    return Operation(state_->collapse_dim_with_dim_minus_one(dim));

}

Operation Operation::transpose(const std::vector<int>& permutation) const {
    return Operation(state_->transpose(permutation));
}

Operation::operator Assignable<Array> () const {
    return state_->operator Assignable<Array>();
}


///////////////////////////////////////////////////////////////////////////////
//                                OP2                                        //
///////////////////////////////////////////////////////////////////////////////




namespace op2 {
    Operation all_reduce(const Operation& a,
                         const std::string& reducer_name) {
        return Operation(std::make_shared<AllReducerOperationState>(
            reducer_name,
            a.state_
        ));
    }

    Operation axis_reduce(const Operation& a,
                          const std::string& reducer_name,
                          const std::vector<int>& axes) {
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
            return all_reduce(a, reducer_name);
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
            res = res.transpose(new_axes_order);
        }
        int num_low_axes_to_reduce = normalized_axes.size();
        if (num_low_axes_to_reduce > 0) {
            int axes_used_up = 0;
            int collapsed_ndim = ndim - 1;
            for (int axes_used_up = 0; axes_used_up < num_low_axes_to_reduce; ++axes_used_up) {
                if (num_low_axes_to_reduce - axes_used_up == 1) {
                    res = Operation(std::make_shared<AxisReducerOperationState>(
                        reducer_name,
                        res.state_
                    ));
                } else {
                    if (res.is_dim_collapsible_with_dim_minus_one(collapsed_ndim)) {
                        res = res.collapse_dim_with_dim_minus_one(collapsed_ndim);
                    } else {
                        res = Operation(std::make_shared<AxisReducerOperationState>(
                            reducer_name,
                            res.state_
                        ));
                    }
                }
                --collapsed_ndim;
            }
        }
        return res;
    }

    Operation argument_all_reduce(const Operation& a,
                                 const std::string& reducer_name) {
        return Operation(std::make_shared<ArgumentAllReducerOperationState>(
            reducer_name,
            a.state_
        ));
    }

    Operation argument_axis_reduce(const Operation& a,
                                   const std::string& reducer_name,
                                   const int& axis) {
        int ndim = a.ndim();
        if (ndim == 0) return Operation(0);
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
            res = res.transpose(axes);
        }
        return Operation(std::make_shared<ArgumentAxisReducerOperationState>(
            reducer_name,
            res.state_
        ));
    }

    Operation elementwise(const Operation& a,
                          const std::string& functor_name) {

        return Operation(std::make_shared<ElementwiseOperationState>(
            functor_name,
            operation_state_ptrs({a.state_})
        ));
    }

    Operation elementwise(
        const Operation& a,
        const Operation& b,
        const std::string& functor_name) {

        // perform type promotion:
        if (a.dtype() != b.dtype()) {
            auto new_type = type_promotion(a, b);
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
            ASSERT2(ndim_compatible(a, b), "ranks don't match");
            auto ret = Operation(std::make_shared<ElementwiseOperationState>(
                functor_name,
                operation_state_ptrs({a.state_, b.state_})
            ));

            // make sure that we fail early if bshapes are not compatible
            ret.bshape();

            return ret;
        }
    }

    // Operation binary_kernel_function(
    //     const Operation& a,
    //     const Operation& b,
    //     const std::string& function_name,
    //     const std::string& kernel_code
    // ) {
    //     // perform type promotion:
    //     if (a.dtype() != b.dtype()) {
    //         auto new_type = Operation::type_promotion(a, b);
    //         if (a.dtype() == new_type) {
    //             // b's dtype is being promoted
    //             return binary_kernel_function(
    //                 a,
    //                 astype(b, new_type),
    //                 function_name,
    //                 kernel_code
    //             );
    //         } else {
    //             // a's dtype is being promoted
    //             return binary_kernel_function(
    //                 astype(a, new_type),
    //                 b,
    //                 function_name,
    //                 kernel_code
    //             );
    //         }
    //     } else {
    //         ASSERT2(Operation::ndim_compatible(a, b), "ranks don't match");
    //         auto output_bshape = get_function_bshape(a, b);

    //         return Operation(
    //             Operation::FUSED_OP_KERNEL_T,
    //             function_name,
    //             kernel_code,
    //             {a, b},
    //             a.dtype()
    //         );
    //     }
    // }

    Operation astype(const Operation& a, DType type) {
        return Operation(std::make_shared<CastOperationState>(
            type,
            a.state_
        ));
    }
}
