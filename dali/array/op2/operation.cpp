#include "operation.h"
#include "dali/config.h"

#include <unordered_set>

#include "dali/array/function2/compiler.h"
#include "dali/array/op2/elementwise_operation.h"
#include "dali/array/op2/gather.h"
#include "dali/array/op2/reducers.h"
#include "dali/array/op2/gather_from_rows.h"
#include "dali/array/op2/rtc_utils.h"
#include "dali/utils/make_message.h"
#include "dali/utils/scope.h"

using utils::Hasher;

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

bool OperationState::is_assignable() const {
    return false;
}

int OperationState::number_of_elements() const {
    return hypercube_volume(shape());
}

void OperationState::full_operation_name(std::stringstream* ss) const {
    *ss << name();
    auto args = arguments();
    if (args.size() > 0) {
        *ss << "(";
        for (int i = 0; i < args.size(); i++) {
            args[i]->full_operation_name(ss);
            if (i + 1 != args.size()) {
                *ss << ", ";
            }
        }
        *ss << ")";
    }
}

std::string OperationState::full_operation_name() const {
    std::stringstream ss;
    full_operation_name(&ss);
    return ss.str();
}

std::string OperationState::prefix_code(const node_to_info_t& node_to_info, memory::DeviceT device_type) const {
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


std::string OperationState::get_code_template(memory::Device device,
                                              const std::vector<const ArrayOperationState*>& arrays,
                                              const std::vector<const ScalarOperationState*>& scalars,
                                              const node_to_info_t& node_to_info) const {
    std::unordered_set<hash_t> prefix_code_visited;
    std::stringstream result;

    this->for_all_suboperations([&](const OperationState* node) {
        auto pc      = node->prefix_code(node_to_info, device.type());
        auto pc_hash = utils::get_hash(pc);
        if (prefix_code_visited.find(pc_hash) == prefix_code_visited.end()) {
            result << pc;
            prefix_code_visited.insert(pc_hash);
        }
    });

    result << "void run(const std::vector<Array>& array_arguments, const std::vector<double>& scalar_arguments, memory::Device device) {\n";

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
    result << get_call_code_nd(symbol_table, node_to_info, device.type());
    result << "}\n";
    return result.str();
}


std::function<void(const std::vector<Array>&, const std::vector<double>&, memory::Device)> OperationState::compile(
        memory::Device device,
        const std::vector<const ArrayOperationState*>& arrays,
        const std::vector<const ScalarOperationState*>& scalars,
        const node_to_info_t& node_to_info) const {
    DALI_SCOPE("get_function");
    // compute a quasi-unique hash for the fused operation
    hash_t hash = Hasher().add((int)device.type())
                          .add(node_to_info.at(this).hash)
                          .value();
    // check if the operation needs to be runtime compiled
    if (!array_op_compiler.load(hash)) {
        DALI_SCOPE("compilation");
        auto code_template = get_code_template(
            device,
            arrays,
            scalars,
            node_to_info
        );
        array_op_compiler.compile<const std::vector<Array>&, const std::vector<double>&, memory::Device>(
            hash,
            code_template,
            device.type()
        );
    }
    // return the operation that was loaded or compiled:
    return array_op_compiler.get_function<const std::vector<Array>&, const std::vector<double>&, memory::Device>(hash);
}

std::string Operation::name() const {
    return state_->full_operation_name();
}

// TODO(jonathan, szymon): move this into member function
void eval_op(const Operation& op,
             const std::vector<int>& output_shape,
             const memory::Device& output_device) {

    auto& self = *op.state_;
    int desired_computation_rank = self.min_computation_rank_;
    std::vector<const ArrayOperationState*> array_ops;
    std::vector<const ScalarOperationState*> scalar_ops;
    OperationState::node_to_info_t node_to_info;

    self.compute_node_compilation_info(desired_computation_rank,
                                       output_shape,
                                       &array_ops,
                                       &scalar_ops,
                                       &node_to_info);

    auto compiled_self = self.compile(output_device,
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

    std::vector<double> scalars;
    std::transform(scalar_ops.begin(),
                   scalar_ops.end(),
                   std::back_inserter(scalars),
                   [&](const ScalarOperationState* op) {
                       return op->value_;
                   });
    std::string name;
    if (Scope::has_observers()) {
        name = op.name();
    }
    DALI_SCOPE(name);
    compiled_self(arrays, scalars, output_device);
}

std::vector<int> get_auto_reduce_axes(const Array& output, const std::vector<int>& in_bshape) {
    if (output.is_stateless() || output.ndim() != in_bshape.size()) {
        return {};
    }

    auto out_bshape = output.bshape();
    std::vector<int> reduction_axes;

    for (int i = 0; i < out_bshape.size(); ++i) {
        if (out_bshape[i] < 0) {
            ASSERT2(out_bshape[i] == -1,
                    "Assigning to broadcast_reshaped Array is not supported.");
            if (std::abs(in_bshape[i]) > 1) {
                // see if number of reductions is greater than 1 or equal to size of output.
                reduction_axes.emplace_back(i);
            }
        }
    }
    return reduction_axes;
}


// TODO(jonathan, szymon): This explicitly locates array ops, it's a bit of a violation
// of contract because, everywhere else, we assume that compute_compilation_info is the
// one responsible for identifying the arrays participating in the expression.
// Hence, we should refactor the code so that we only collect args in one place.

// returns device_proposal, device_found (if not args are present it's hard to suggest anything)
std::tuple<memory::Device, bool> OperationState::preferred_device() const {
    int args_read = 0;
    bool shared_common_device = true;
    memory::Device common_preferred_device;
    memory::Device output_device;

    for_all_suboperations([&](const OperationState* op) {
        auto ptr = dynamic_cast<const ArrayOperationState*>(op);


        if (ptr != NULL) {
            auto mem = ptr->array_.memory();

            // When state args_read <= 0, then reduction is in its first Array argument
            // while other non-Array arguments have been ignored by ReduceOverArgs<>::reduce_helper
            // [Note: output is also an Array argument]
            if (args_read <= 0) {
                // *** When considering the first Array ***
                // If there's only 1 Array involved, we can safely consider
                // this Array's memory's preferred_device as a good option
                output_device = mem->preferred_device;
                // One caveat, we want preferred_device's memory to be fresh
                bool is_best_option_fresh = mem->is_fresh(mem->preferred_device);
                // Also we want to know whether any copy of memory is fresh
                bool is_some_other_option_fresh = mem->is_any_fresh();
                // if the preferred memory is not fresh, and there is
                // a fresh alternative use it:
                if (!is_best_option_fresh && is_some_other_option_fresh) {
                    output_device = mem->find_some_fresh_device();
                }// else, make the preferred device fresh

                common_preferred_device = mem->preferred_device;
            } else {
                if (mem->preferred_device != common_preferred_device || !shared_common_device) {
                    // When considering other arguments, if the next argument prefers a different device,
                    // then we fallback to the tie-breaker device
                    output_device = memory::default_preferred_device;
                    shared_common_device = false;
                } else {
                    // we can place the computation on the currently agreed device
                }
            }
            ++args_read;
        }
    });

    if (args_read == 0) {
        return std::make_tuple(memory::default_preferred_device, false);
    } else {
        return std::make_tuple(output_device, true);
    }

}

OperationState::operator Assignable<Array> () const {
    auto this_ptr = shared_from_this();
    return Assignable<Array>([this_ptr](Array& out, const OPERATOR_T& operator_t) mutable {
        auto output_dtype_proposal  = this_ptr->dtype();
        auto output_bshape_proposal = this_ptr->bshape();

        bool device_found;
        memory::Device output_device_proposal;
        std::tie(output_device_proposal, device_found) = this_ptr->preferred_device();

        auto op = Operation(this_ptr);
        Array out_array;


        OPERATOR_T operator_to_use = operator_t == OPERATOR_T_LSE ? OPERATOR_T_ADD : operator_t;

        if (operator_t == OPERATOR_T_LSE) {
            std::vector<int> reduction_axes = get_auto_reduce_axes(out, output_bshape_proposal);
            if (reduction_axes.size() > 0) {
                op = op::sum(op, reduction_axes);
                // add the reduced dimensions back:
                for (int i = 0; i < reduction_axes.size(); ++i) {
                    output_bshape_proposal[reduction_axes[i]] = 1;
                }
            }
            initialize_output_array(
                out,
                output_dtype_proposal,
                output_device_proposal,
                &output_bshape_proposal
            );
            out_array = out;
            for (int i = int(reduction_axes.size()) - 1; i >= 0; --i) {
                out_array = out_array.squeeze(reduction_axes[i]);
            }
            if (reduction_axes.size() > 0) {
                output_bshape_proposal = op.bshape();
            }
        } else {
            initialize_output_array(
                out,
                output_dtype_proposal,
                output_device_proposal,
                &output_bshape_proposal
            );
            out_array = out;
        }

        if (!out.memory()->is_any_fresh() && operator_to_use == OPERATOR_T_ADD) {
            // if operation is += to an empty/zeros array, then switch operator
            // to equal:
            operator_to_use = OPERATOR_T_EQL;
        }

        memory::Device output_device;
        if (device_found) {
            if (out_array.memory()->preferred_device != output_device_proposal) {
                output_device = memory::default_preferred_device;
            } else {
                output_device = output_device_proposal;
            }
        } else {
            output_device = out_array.memory()->preferred_device;
        }

        auto self_op = op::assign(out_array, operator_to_use, op);
        eval_op(self_op, self_op.shape(), output_device);
    });
}

OperationState::operator Assignable<ArrayGather> () const {
    auto this_ptr = shared_from_this();
    return Assignable<ArrayGather>([this_ptr](ArrayGather& out, const OPERATOR_T& operator_t) mutable {
        auto output_dtype  = out.dtype();
        auto output_device = memory::Device::cpu();
        auto self_op = op::assign(op::gather(out.source, out.indices), operator_t == OPERATOR_T_LSE ? OPERATOR_T_ADD : operator_t, Operation(this_ptr));
        eval_op(self_op, self_op.shape(), output_device);
    });
}

OperationState::operator Assignable<ArraySubtensor> () const {
    auto this_ptr = shared_from_this();
    return Assignable<ArraySubtensor>([this_ptr](ArraySubtensor& out, const OPERATOR_T& operator_t) mutable {
        auto output_dtype  = out.dtype();
        auto output_device = memory::Device::cpu();
        auto self_op = op::assign(op::gather_from_rows(out.source, out.indices), operator_t == OPERATOR_T_LSE ? OPERATOR_T_ADD : operator_t, Operation(this_ptr));
        eval_op(self_op, self_op.shape(), output_device);
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

std::string ArrayOperationState::name() const {
    return "Array";
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

bool ArrayOperationState::is_assignable() const {
    return true;
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
                                         .add(contiguous())
                                         .add(array_.dtype()).value();
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

std::string ArrayOperationState::get_call_code_nd(const symbol_table_t& symbol_table, const node_to_info_t& node_to_info, memory::DeviceT device_type) const {
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

std::string ScalarOperationState::name() const {
    return "double";
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

std::string ScalarOperationState::get_call_code_nd(const symbol_table_t& symbol_table, const node_to_info_t& node_to_info, memory::DeviceT device_type) const {
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

Operation::Operation(int scalar): Operation(op::astype((double)scalar, DTYPE_INT32)) {
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

std::vector<int> Operation::shape() const {
    return state_->shape();
}

bool Operation::is_scalar() const {
    return ndim() == 0;
}

bool Operation::is_assignable() const {
    return state_->is_assignable();
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
Operation::operator Assignable<ArrayGather> () const {
    return state_->operator Assignable<ArrayGather>();
}
Operation::operator Assignable<ArraySubtensor> () const {
    return state_->operator Assignable<ArraySubtensor>();
}

struct AssignmentOperationState : public OperationState {
    static const hash_t optype_hash;
    operation_state_ptr left_;
    operation_state_ptr right_;
    OPERATOR_T operator_t_;

    AssignmentOperationState(operation_state_ptr left,
                             const OPERATOR_T& operator_t,
                             operation_state_ptr right) :
            OperationState(std::max(left->min_computation_rank_, right->min_computation_rank_)),
            left_(left), right_(right), operator_t_(operator_t) {
    }


    virtual std::string prefix_code(
            const node_to_info_t& node_to_info,
            memory::DeviceT device_type) const {
        if (device_type == memory::DEVICE_T_GPU) {
            if (node_to_info.at(this).computation_rank == 1) {
                return utils::make_message(
                    "template<typename Destination, typename Source>\n"
                    "void __global__\n"
                    "assign_kernel(Destination dst, Source src, int num_el) {\n"
                    "    int idx = blockDim.x * blockIdx.x + threadIdx.x;\n"
                    "    int stride = blockDim.x * gridDim.x;\n"
                    "    for (int i = idx; i < num_el; i += stride) {\n"
                    "        dst(i) ", operator_to_name(operator_t_), " src(i);\n"
                    "    }\n"
                    "}\n"
                );
            } else {
                ASSERT2(false, "not implemented");
            }
        }

    }

    virtual bool is_assignable() const {
        return false;
    }

    virtual DType dtype() const {
        return left_->dtype();
    }

    virtual std::string name() const {
        return "assign";
    }

    virtual void full_operation_name(std::stringstream* ss) const {
        left_->full_operation_name(ss);
        (*ss) << " " << operator_to_name(operator_t_) << " ";
        right_->full_operation_name(ss);
    }

    virtual void compute_node_compilation_info(int desired_computation_rank,
                                               const std::vector<int>& desired_computation_shape,
                                               std::vector<const ArrayOperationState*>* arrays,
                                               std::vector<const ScalarOperationState*>* scalars,
                                               node_to_info_t* node_to_info) const {
        (*node_to_info)[this].computation_rank  = desired_computation_rank;
        (*node_to_info)[this].computation_shape = desired_computation_shape;
        left_->compute_node_compilation_info(desired_computation_rank, desired_computation_shape, arrays, scalars, node_to_info);
        right_->compute_node_compilation_info(desired_computation_rank, desired_computation_shape, arrays, scalars, node_to_info);
        (*node_to_info)[this].hash = Hasher().add(operator_t_)
                                             .add(optype_hash)
                                             .add(node_to_info->at(left_.get()).hash)
                                             .add(node_to_info->at(right_.get()).hash)
                                             .add(desired_computation_rank).value();
    }

    virtual int ndim() const {
        return left_->ndim();
    }

    virtual std::vector<int> bshape() const {
        return left_->bshape();
    }

    virtual operation_state_ptrs arguments() const {
        return {left_, right_};
    }

    virtual std::string assignment_code(const symbol_table_t& symbol_table,
                                        const node_to_info_t& node_to_info,
                                        memory::DeviceT device_type) const {
        int computation_rank = node_to_info.at(this).computation_rank;
        std::string indexing_nd = computation_rank == 1 ? "(i)" : "[" + generate_accessor_string(computation_rank) + "]";
        return utils::make_message(
            left_->get_call_code_nd(symbol_table, node_to_info, device_type), indexing_nd, " ",
            operator_to_name(operator_t_),
            " ",
            right_->get_call_code_nd(symbol_table, node_to_info, device_type), indexing_nd, ";\n"
        );
    }

    virtual std::string get_call_code_nd(const symbol_table_t& symbol_table,
                                         const node_to_info_t& node_to_info,
                                         memory::DeviceT device_type) const {
        if (device_type == memory::DEVICE_T_CPU) {
            // TODO: debate if we want to allow chaining here:
            //       (e.g. call this assignment, then this assignment, etc...)
            int computation_rank = node_to_info.at(this).computation_rank;
            if (computation_rank == 1) {
                return utils::make_message(
                    "    int num_el = ", left_->get_call_code_nd(symbol_table, node_to_info, device_type), ".shape().numel();\n",
                    "    #pragma clang loop vectorize(enable)\n",
                    "    #pragma clang loop interleave(enable)\n",
                    "    for (int i = 0; i < num_el; ++i) {\n",
                    "        ", assignment_code(symbol_table, node_to_info, device_type),
                    "    }\n"
                );
            } else {
                return construct_for_loop(
                    computation_rank,
                    assignment_code(symbol_table, node_to_info, device_type),
                    left_->get_call_code_nd(symbol_table, node_to_info, device_type),
                    4
                );
            }
        }
#ifdef DALI_USE_CUDA
        else if (device_type == memory::DEVICE_T_GPU) {
            return utils::make_message(
                    "    int num_el = ", left_->get_call_code_nd(symbol_table, node_to_info, device_type), ".shape().numel();\n"
                    "    const int NT = 128;\n"
                    "    const int MAX_BLOCKS = 40960;\n"
                    "    int grid_size = div_ceil(num_el, NT);\n"
                    "    assert(grid_size <= MAX_BLOCKS);\n"
                    "    assign_kernel<<<grid_size, NT, 0, NULL>>>(\n"
                    "        ", left_->get_call_code_nd(symbol_table, node_to_info, device_type), ",\n"
                    "        ", right_->get_call_code_nd(symbol_table, node_to_info, device_type), ",\n"
                    "        num_el\n"
                    "    );\n"
            );
        }
#endif
        else {
            ASSERT2(false, "unknown device type.");
        }

    }
};
const hash_t AssignmentOperationState::optype_hash = std::hash<std::string>()("AssignmentOperationState");


namespace op {
    Operation assign(const Operation& left, const OPERATOR_T& operator_t, const Operation& right) {
        ASSERT2(left.is_assignable(), "left Operation of assign must be assignable.");
        return Operation(std::make_shared<AssignmentOperationState>(left.state_, operator_t, right.state_));
    }
}
