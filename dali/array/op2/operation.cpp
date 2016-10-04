#include "operation.h"

#include <unordered_set>

#include "dali/array/function2/compiler.h"
#include "dali/array/op2/elementwise_operation.h"
#include "dali/array/op2/rtc_utils.h"
#include "dali/utils/make_message.h"

using utils::Hasher;

///////////////////////////////////////////////////////////////////////////////
//                         MISC UTILS                                        //
///////////////////////////////////////////////////////////////////////////////

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

int OperationState::ndim() const {
    return bshape().size();
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

