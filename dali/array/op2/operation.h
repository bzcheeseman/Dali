#ifndef DALI_ARRAY_OP2_OPERATION_H
#define DALI_ARRAY_OP2_OPERATION_H

#include <memory>
#include <vector>
#include <unordered_map>
#include <tuple>

#include "dali/array/array.h"
#include "dali/utils/hash_utils.h"


struct OperationState;
struct LValueOperationState;
struct RValueOperationState;
struct RunnableOperationState;
struct ArrayOperationState;
struct ScalarOperationState;
struct JITOperationState;

typedef std::shared_ptr<const OperationState>  operation_state_ptr;
typedef std::vector<operation_state_ptr>       operation_state_ptrs;

struct CompilationInfo {
    int    computation_rank;
    std::string name;
    std::vector<int> computation_shape;
    hash_t hash;
};

struct OperationState : std::enable_shared_from_this<OperationState> {
    ///////////////////////////////////////////////////////////////////////////////
    //            MUST REIMPLEMENT FUNCTIONS BELOW                               //
    ///////////////////////////////////////////////////////////////////////////////
    virtual DType dtype() const = 0;
    virtual std::vector<int> bshape() const = 0;
    virtual std::string name() const = 0;

    ///////////////////////////////////////////////////////////////////////////////
    //            REIMPLEMENT AS YOU SEE FIT                                     //
    ///////////////////////////////////////////////////////////////////////////////

    virtual int ndim() const;

    virtual std::vector<int> shape() const;

    virtual int number_of_elements() const;

    virtual std::vector<operation_state_ptr> arguments() const;

    // should almost never be reimplemented:
    virtual void full_operation_name(std::stringstream*) const;

    // returns device_proposal, device_found (if no args are present it's hard to suggest anything)
    virtual std::tuple<memory::Device, bool> preferred_device() const;

    ///////////////////////////////////////////////////////////////////////////////
    //            DO NOT REIMPLEMENT FUNCTIONS BELOW                             //
    ///////////////////////////////////////////////////////////////////////////////
    virtual std::string full_operation_name() const;

    virtual std::shared_ptr<const LValueOperationState> as_lvalue()   const;
    virtual std::shared_ptr<const RValueOperationState> as_rvalue()   const;
    virtual std::shared_ptr<const JITOperationState>    as_jit()   const;
    virtual std::shared_ptr<const ArrayOperationState>  as_array()   const;

    operator Assignable<Array> () const;
    operator Assignable<ArrayGather> () const;
    operator Assignable<ArraySubtensor> () const;

    virtual void for_all_suboperations(std::function<void(const OperationState*)> callback) const final;
};



struct RValueOperationState: virtual public OperationState {
    virtual std::shared_ptr<const RunnableOperationState> assign_to(std::shared_ptr<const LValueOperationState> op, memory::Device device) const = 0;
    virtual std::shared_ptr<const RunnableOperationState> add_to(std::shared_ptr<const LValueOperationState> op, memory::Device device) const;
    virtual std::shared_ptr<const RunnableOperationState> sub_to(std::shared_ptr<const LValueOperationState> op, memory::Device device) const;
    virtual std::shared_ptr<const RunnableOperationState> mul_to(std::shared_ptr<const LValueOperationState> op, memory::Device device) const;
    virtual std::shared_ptr<const RunnableOperationState> div_to(std::shared_ptr<const LValueOperationState> op, memory::Device device) const;

    virtual std::shared_ptr<const RunnableOperationState> as_runnable(memory::Device device) const;
    virtual std::shared_ptr<const ArrayOperationState>    initialize_destination(memory::Device device) const;

    virtual std::shared_ptr<const RunnableOperationState> operator_to(OPERATOR_T operator_t, std::shared_ptr<const LValueOperationState> op, memory::Device device) const final;
};

struct LValueOperationState: virtual public OperationState {
    // assumes that op,destination_array() does not return NULL.
    virtual std::shared_ptr<const RunnableOperationState> assign_from(std::shared_ptr<const RunnableOperationState> op, memory::Device device) const = 0;
    virtual std::shared_ptr<const RunnableOperationState> add_from(std::shared_ptr<const RunnableOperationState> op, memory::Device device) const = 0;
    virtual std::shared_ptr<const RunnableOperationState> sub_from(std::shared_ptr<const RunnableOperationState> op, memory::Device device) const = 0;
    virtual std::shared_ptr<const RunnableOperationState> mul_from(std::shared_ptr<const RunnableOperationState> op, memory::Device device) const = 0;
    virtual std::shared_ptr<const RunnableOperationState> div_from(std::shared_ptr<const RunnableOperationState> op, memory::Device device) const = 0;

    virtual std::shared_ptr<const RunnableOperationState> operator_from(OPERATOR_T operator_t, std::shared_ptr<const RunnableOperationState> op, memory::Device device) const final;

};

struct LRValueOperationState: public LValueOperationState, public RValueOperationState {
    virtual std::shared_ptr<const RunnableOperationState> add_from(std::shared_ptr<const RunnableOperationState> op, memory::Device device) const;
    virtual std::shared_ptr<const RunnableOperationState> sub_from(std::shared_ptr<const RunnableOperationState> op, memory::Device device) const;
    virtual std::shared_ptr<const RunnableOperationState> mul_from(std::shared_ptr<const RunnableOperationState> op, memory::Device device) const;
    virtual std::shared_ptr<const RunnableOperationState> div_from(std::shared_ptr<const RunnableOperationState> op, memory::Device device) const;
};


struct RunnableOperationState : virtual public RValueOperationState {
    virtual void run() const = 0;
    // if and only if this operation is assignment to an array, return operation corresponding
    // to that array. Otherwise return NULL.
    virtual std::shared_ptr<const OperationState> destination_op() const = 0;

    virtual std::shared_ptr<const RunnableOperationState> assign_to(std::shared_ptr<const LValueOperationState> op, memory::Device device) const;
    virtual std::shared_ptr<const RunnableOperationState> add_to(std::shared_ptr<const LValueOperationState> op, memory::Device device) const;
    virtual std::shared_ptr<const RunnableOperationState> sub_to(std::shared_ptr<const LValueOperationState> op, memory::Device device) const;
    virtual std::shared_ptr<const RunnableOperationState> mul_to(std::shared_ptr<const LValueOperationState> op, memory::Device device) const;
    virtual std::shared_ptr<const RunnableOperationState> div_to(std::shared_ptr<const LValueOperationState> op, memory::Device device) const;

};





struct JITOperationState : OperationState {
    typedef std::unordered_map<const JITOperationState*, std::string>     symbol_table_t;
    typedef std::unordered_map<const JITOperationState*, CompilationInfo> node_to_info_t;

    const int min_computation_rank_;

    JITOperationState() = delete;
    JITOperationState(int min_computation_rank);

    virtual void compute_node_compilation_info(int desired_computation_rank,
                                               const std::vector<int>& desired_computation_shape,
                                               std::vector<const ArrayOperationState*>* arrays,
                                               std::vector<const ScalarOperationState*>* scalars,
                                               node_to_info_t* node_to_info) const = 0;

    virtual std::string get_call_code_nd(const symbol_table_t& symbol_table, const node_to_info_t& node_to_info, memory::DeviceT device_type) const = 0;


    ///////////////////////////////////////////////////////////////////////////////
    //            REIMPLEMENT AS YOU SEE FIT                                     //
    ///////////////////////////////////////////////////////////////////////////////

    virtual std::string prefix_code(const node_to_info_t& node_to_info, memory::DeviceT device_type) const;


    // Returns true if striding is such that dim and (dim - 1) can be merged into
    // single dim. This function is allowed to returns false negatives, so if
    // your case is really complicated just return false (bear in mind that this
    // might sacrifice efficieny).
    virtual bool is_dim_collapsible_with_dim_minus_one(const int& dim) const;

    virtual std::shared_ptr<const JITOperationState> collapse_dim_with_dim_minus_one(const int& dim) const;

    virtual std::shared_ptr<const JITOperationState> transpose(const std::vector<int>& permutation) const;

    virtual bool is_assignable() const;

    ///////////////////////////////////////////////////////////////////////////////
    //            DO NOT REIMPLEMENT FUNCTIONS BELOW                             //
    ///////////////////////////////////////////////////////////////////////////////


    virtual std::string get_code_template(memory::Device device,
                                          const std::vector<const ArrayOperationState*>& arrays,
                                          const std::vector<const ScalarOperationState*>& scalars,
                                          const node_to_info_t& node_to_info) const final;


    std::function<void(void**, const int*, const int**, const int**, const void**)> compile(
            memory::Device device,
            const std::vector<const ArrayOperationState*>& arrays,
            const std::vector<const ScalarOperationState*>& scalars,
            const node_to_info_t& node_to_info) const;

    virtual std::shared_ptr<const JITOperationState> jit_shared_from_this() const final;
    virtual std::shared_ptr<JITOperationState> jit_shared_from_this() final;
};


struct ArrayOperationState : public LRValueOperationState {
    static const hash_t optype_hash;

    Array array_;

    ArrayOperationState(Array array);

    virtual DType dtype() const;

    virtual std::vector<int> bshape() const;

    virtual int ndim() const;
    virtual std::string name() const;
    virtual bool is_assignable() const;

    virtual bool contiguous() const;

    virtual std::vector<int> shape() const;

    virtual int number_of_elements() const;

    // virtual void compute_node_compilation_info(int desired_computation_rank,
    //                                            const std::vector<int>& desired_computation_shape,
    //                                            std::vector<const ArrayOperationState*>* arrays,
    //                                            std::vector<const ScalarOperationState*>* scalars,
    //                                            node_to_info_t* node_to_info) const;

    // virtual bool is_dim_collapsible_with_dim_minus_one(const int& dim) const;

    // virtual operation_state_ptr collapse_dim_with_dim_minus_one(const int& dim) const;

    // virtual operation_state_ptr transpose(const std::vector<int>& permutation) const;

    // virtual std::string get_call_code_nd(const symbol_table_t& symbol_table, const node_to_info_t& node_to_info, memory::DeviceT device_type) const;

    virtual std::shared_ptr<const RunnableOperationState> assign_to(std::shared_ptr<const LValueOperationState> op, memory::Device device) const;
    virtual std::shared_ptr<const RunnableOperationState> add_to(std::shared_ptr<const LValueOperationState> op, memory::Device device) const;
    virtual std::shared_ptr<const RunnableOperationState> sub_to(std::shared_ptr<const LValueOperationState> op, memory::Device device) const;
    virtual std::shared_ptr<const RunnableOperationState> mul_to(std::shared_ptr<const LValueOperationState> op, memory::Device device) const;
    virtual std::shared_ptr<const RunnableOperationState> div_to(std::shared_ptr<const LValueOperationState> op, memory::Device device) const;

    virtual std::shared_ptr<const RunnableOperationState> assign_from(std::shared_ptr<const RunnableOperationState> op, memory::Device device) const;
    virtual std::shared_ptr<const RunnableOperationState> add_from(std::shared_ptr<const RunnableOperationState> op, memory::Device device) const;
    virtual std::shared_ptr<const RunnableOperationState> sub_from(std::shared_ptr<const RunnableOperationState> op, memory::Device device) const;
    virtual std::shared_ptr<const RunnableOperationState> mul_from(std::shared_ptr<const RunnableOperationState> op, memory::Device device) const;
    virtual std::shared_ptr<const RunnableOperationState> div_from(std::shared_ptr<const RunnableOperationState> op, memory::Device device) const;
};


struct ScalarOperationState : public JITOperationState {
    ScalarOperationState();
    static const hash_t optype_hash;

    virtual DType dtype() const = 0;
    virtual std::vector<int> bshape() const;

    virtual int ndim() const;
    virtual std::string name() const = 0;
    virtual const void* value_ptr() const = 0;

    virtual std::vector<int> shape() const;

    virtual int number_of_elements() const;

    virtual void compute_node_compilation_info(int desired_computation_rank,
                                               const std::vector<int>& desired_computation_shape,
                                               std::vector<const ArrayOperationState*>* arrays,
                                               std::vector<const ScalarOperationState*>* scalars,
                                               node_to_info_t* node_to_info) const;

    virtual bool is_dim_collapsible_with_dim_minus_one(const int& dim) const;

    virtual std::shared_ptr<const JITOperationState> transpose(const std::vector<int>& permutation) const;

    virtual std::string get_call_code_nd(const symbol_table_t& symbol_table, const node_to_info_t& node_to_info, memory::DeviceT device_type) const;
};



struct AbstractAssignOperationState : public RValueOperationState {
    std::shared_ptr<const LValueOperationState> left_;
    std::shared_ptr<const RValueOperationState> right_;
    OPERATOR_T operator_t_;

    AbstractAssignOperationState(std::shared_ptr<const LValueOperationState>  left,
                                 const OPERATOR_T& operator_t,
                                 std::shared_ptr<const RValueOperationState>  right);
    virtual DType dtype() const;

    virtual std::string name() const;

    virtual void full_operation_name(std::stringstream* ss) const;

    virtual bool is_assignable() const;

    virtual int ndim() const;

    virtual std::vector<int> bshape() const;

    virtual operation_state_ptrs arguments() const;

    virtual std::shared_ptr<const RunnableOperationState> assign_to(std::shared_ptr<const LValueOperationState> op, memory::Device device) const;
    virtual std::shared_ptr<const RunnableOperationState> add_to(std::shared_ptr<const LValueOperationState> op, memory::Device device) const;
    virtual std::shared_ptr<const RunnableOperationState> sub_to(std::shared_ptr<const LValueOperationState> op, memory::Device device) const;
    virtual std::shared_ptr<const RunnableOperationState> mul_to(std::shared_ptr<const LValueOperationState> op, memory::Device device) const;
    virtual std::shared_ptr<const RunnableOperationState> div_to(std::shared_ptr<const LValueOperationState> op, memory::Device device) const;


    virtual std::shared_ptr<const RunnableOperationState> as_runnable(memory::Device device) const;
    virtual std::shared_ptr<const ArrayOperationState>    initialize_destination(memory::Device device) const;
};

struct Operation {
    operation_state_ptr state_;

    Operation(const Array& arr);

    Operation(const Assignable<Array>& arr);

    Operation(double scalar);
    Operation(float scalar);
    Operation(int scalar);

    Operation(operation_state_ptr state);

    DType dtype() const;

    int ndim() const;

    bool is_scalar() const;
    bool is_assignable() const;

    std::vector<int> bshape() const;
    std::vector<int> shape() const;
    std::string name() const;

    int number_of_elements() const;

    operator Assignable<Array>() const;
    operator Assignable<ArrayGather>() const;
    operator Assignable<ArraySubtensor>() const;
};

namespace op {
    Operation assign(const Operation& left, const OPERATOR_T& operator_t, const Operation& right);
}  // namespace op

#endif  // DALI_ARRAY_OP2_OPERATION_H
