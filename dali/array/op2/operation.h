#ifndef DALI_ARRAY_OP2_OPERATION_H
#define DALI_ARRAY_OP2_OPERATION_H

#include <memory>
#include <vector>
#include <unordered_map>
#include <tuple>

#include "dali/array/array.h"
#include "dali/utils/hash_utils.h"


struct OperationState;
struct ArrayOperationState;
struct ScalarOperationState;

typedef std::shared_ptr<const OperationState>  operation_state_ptr;
typedef std::vector<operation_state_ptr>       operation_state_ptrs;

struct CompilationInfo {
    int    computation_rank;
    std::vector<int> computation_shape;
    hash_t hash;
};

struct OperationState : std::enable_shared_from_this<OperationState> {
    typedef std::unordered_map<const OperationState*, std::string>     symbol_table_t;
    typedef std::unordered_map<const OperationState*, CompilationInfo> node_to_info_t;

    const int min_computation_rank_;

    ///////////////////////////////////////////////////////////////////////////////
    //            MUST REIMPLEMENT FUNCTIONS BELOW                               //
    ///////////////////////////////////////////////////////////////////////////////

    virtual DType dtype() const = 0;
    virtual std::vector<int> bshape() const = 0;



    virtual void compute_node_compilation_info(int desired_computation_rank,
                                               const std::vector<int>& desired_computation_shape,
                                               std::vector<const ArrayOperationState*>* arrays,
                                               std::vector<const ScalarOperationState*>* scalars,
                                               node_to_info_t* node_to_info) const = 0;





    virtual std::string get_call_code_nd(const symbol_table_t& symbol_table, const node_to_info_t& node_to_info) const = 0;



    ///////////////////////////////////////////////////////////////////////////////
    //            REIMPLEMENT AS YOU SEE FIT                                     //
    ///////////////////////////////////////////////////////////////////////////////

    virtual int ndim() const;

    virtual std::vector<int> shape() const;

    virtual int number_of_elements() const;

    virtual std::string prefix_code(const node_to_info_t& node_to_info) const;


    virtual std::vector<operation_state_ptr> arguments() const;

    // Returns true if striding is such that dim and (dim - 1) can be merged into
    // single dim. This function is allowed to returns false negatives, so if
    // your case is really complicated just return false (bear in mind that this
    // might sacrifice efficieny).
    virtual bool is_dim_collapsible_with_dim_minus_one(const int& dim) const;

    virtual operation_state_ptr collapse_dim_with_dim_minus_one(const int& dim) const;

    virtual operation_state_ptr transpose(const std::vector<int>& permutation) const;

    OperationState(int min_computation_rank);

    ///////////////////////////////////////////////////////////////////////////////
    //            DO NOT REIMPLEMENT FUNCTIONS BELOW                             //
    ///////////////////////////////////////////////////////////////////////////////

    OperationState() = delete;


    virtual std::string get_code_template(const OPERATOR_T& operator_t,
                                          bool dst_contiguous,
                                          DType output_dtype,
                                          memory::Device device,
                                          int desired_computation_rank,
                                          const std::vector<const ArrayOperationState*>& arrays,
                                          const std::vector<const ScalarOperationState*>& scalars,
                                          const node_to_info_t& node_to_info) const final;

    virtual std::string get_assign_code_nd(const OPERATOR_T& operator_t,
                                           const std::string& call_nd,
                                           const symbol_table_t& symbol_table,
                                           const node_to_info_t& node_to_info) const final;


    std::function<void(Array&, const std::vector<Array>&, const std::vector<double>&)> compile(
            const OPERATOR_T& operator_t,
            bool dst_contiguous,
            DType output_dtype,
            int desired_computation_rank,
            memory::Device device,
            const std::vector<const ArrayOperationState*>& arrays,
            const std::vector<const ScalarOperationState*>& scalars,
            const node_to_info_t& node_to_info) const;


    operator Assignable<Array> () const;


    virtual void for_all_suboperations(std::function<void(const OperationState*)> callback) const final;
};


struct ArrayOperationState : public OperationState {
    static const hash_t optype_hash;

    Array array_;

    ArrayOperationState(Array array);

    virtual DType dtype() const;

    virtual std::vector<int> bshape() const;

    virtual int ndim() const;

    virtual bool contiguous() const;

    virtual std::vector<int> shape() const;

    virtual int number_of_elements() const;

    virtual void compute_node_compilation_info(int desired_computation_rank,
                                               const std::vector<int>& desired_computation_shape,
                                               std::vector<const ArrayOperationState*>* arrays,
                                               std::vector<const ScalarOperationState*>* scalars,
                                               node_to_info_t* node_to_info) const;

    virtual bool is_dim_collapsible_with_dim_minus_one(const int& dim) const;

    virtual operation_state_ptr collapse_dim_with_dim_minus_one(const int& dim) const;

    virtual operation_state_ptr transpose(const std::vector<int>& permutation) const;

    virtual std::string get_call_code_nd(const symbol_table_t& symbol_table, const node_to_info_t& node_to_info) const;
};



struct ScalarOperationState : public OperationState{
    static const hash_t optype_hash;

    double value_;

    ScalarOperationState(double value);

    virtual DType dtype() const;

    virtual std::vector<int> bshape() const;

    virtual int ndim() const;

    virtual std::vector<int> shape() const;

    virtual int number_of_elements() const;

    virtual void compute_node_compilation_info(int desired_computation_rank,
                                               const std::vector<int>& desired_computation_shape,
                                               std::vector<const ArrayOperationState*>* arrays,
                                               std::vector<const ScalarOperationState*>* scalars,
                                               node_to_info_t* node_to_info) const;

    virtual bool is_dim_collapsible_with_dim_minus_one(const int& dim) const;

    virtual operation_state_ptr transpose(const std::vector<int>& permutation) const;

    virtual std::string get_call_code_nd(const symbol_table_t& symbol_table, const node_to_info_t& node_to_info) const;
};



struct ElementwiseOperationState : public OperationState {
    static const hash_t optype_hash;

    const operation_state_ptrs arguments_;
    const std::string functor_name_;

    static int compute_min_computation_rank(const operation_state_ptrs& arguments);

    ElementwiseOperationState(const std::string& functor_name, const operation_state_ptrs& arguments);


    virtual DType dtype() const;

    virtual std::vector<int> bshape() const;

    virtual std::vector<operation_state_ptr> arguments() const;

    virtual void compute_node_compilation_info(int desired_computation_rank,
                                               const std::vector<int>& desired_computation_shape,
                                               std::vector<const ArrayOperationState*>* arrays,
                                               std::vector<const ScalarOperationState*>* scalars,
                                               node_to_info_t* node_to_info) const;

    virtual bool is_dim_collapsible_with_dim_minus_one(const int& dim) const;

    virtual operation_state_ptr collapse_dim_with_dim_minus_one(const int& dim) const;

    virtual operation_state_ptr transpose(const std::vector<int>& permutation) const;

    virtual std::string get_call_code_nd(const symbol_table_t& symbol_table, const node_to_info_t& node_to_info) const;

    virtual std::string prefix_code(const node_to_info_t& node_to_info) const;
};

struct CastOperationState : public ElementwiseOperationState {
    static const hash_t optype_hash;

    const DType dtype_;

    CastOperationState(DType dtype, const operation_state_ptr argument);

    virtual DType dtype() const;

    virtual void compute_node_compilation_info(
        int desired_computation_rank,
        const std::vector<int>& desired_computation_shape,
        std::vector<const ArrayOperationState*>* arrays,
        std::vector<const ScalarOperationState*>* scalars,
        node_to_info_t* node_to_info) const;
};


struct Operation {
    operation_state_ptr state_;

    Operation(const Array& arr);

    Operation(const Assignable<Array>& arr);

    Operation(double scalar);

    Operation(int scalar);

    Operation(operation_state_ptr state);

    DType dtype() const;

    int ndim() const;

    bool is_scalar() const;

    std::vector<int> bshape() const;

    int number_of_elements() const;

    bool is_dim_collapsible_with_dim_minus_one(const int& dim) const;

    Operation collapse_dim_with_dim_minus_one(const int& dim) const;

    Operation transpose(const std::vector<int>& permutation) const;

    operator Assignable<Array>() const;
};


namespace op2 {
    // elementwise kernel given by name.
    // will assume that return type of kernel
    // is given by the `dtype` argument.
    Operation elementwise(
        const Operation& a,
        const std::string& functor_name,
        DType dtype
    );

    // elementwise kernel given by name. assumes
    // return type is unchanged from a's
    Operation elementwise(
        const Operation& a,
        const std::string& functor_name
    );

    // pair-wise kernel. Will type promote arguments
    // so that they have the same type when
    // given to the functor:
    // - float w/. double => double
    // - float w/. int => float
    // - double w/. int => double
    Operation elementwise(
        const Operation& a,
        const Operation& b,
        const std::string& functor_name
    );

    // call a kernel on a pair of arguments. Assumes
    // both arguments should be of the same type. Peforms
    // type promotion on the arguments if not. Will paste
    // and run the associated code `kernel_code` during
    // compilation and usage. (Warning: this might cause
    // collisions when a name is used multiple times)
    Operation binary_kernel_function(
        const Operation& a,
        const Operation& b,
        const std::string& function_name,
        const std::string& kernel_code
    );

    // Perform a type conversion by casting the values in x
    // to another dtype.
    Operation astype(const Operation& x, DType dtype);


    // type-promote arguments if necessary and check whether their
    // ranks are compatible (equal or one is a scalar)
    std::tuple<Operation, Operation> ensure_arguments_compatible(
        const Operation& a, const Operation& b
    );
} // namespace op2

#endif  // DALI_ARRAY_OP2_OPERATION_H
