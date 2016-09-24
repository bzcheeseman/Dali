#ifndef DALI_ARRAY_OP2_COMBINABLE_H
#define DALI_ARRAY_OP2_COMBINABLE_H

#include <vector>
#include "dali/array/array.h"

class FusedOperation {
    public:
        enum FUSED_OP_T {
            FUSED_OP_ARRAY_T = 0,
            FUSED_OP_SCALAR_T = 1,
            FUSED_OP_ELEMENTWISE_T = 2,
            FUSED_OP_KERNEL_T = 3,
            FUSED_OP_ALLREDUCE_T = 4
        };

        // A fused operation taking only an array is no-op
        FusedOperation(const Array& arr);
        FusedOperation(Array&& arr);
        FusedOperation(const Assignable<Array>& arr);
        FusedOperation(const double& scalar);
        FusedOperation(const int& scalar);
        // A fused operation can be constructed using a type identifier
        // 0 -> no-op holding an Array
        // 1 -> no-op holding a scalar double
        // 2 -> elementwise pair kernel
        // 3 -> binary_kernel
        FusedOperation(
            FUSED_OP_T type,
            const std::string& functor_name,
            const std::vector<FusedOperation>& arguments,
            DType dtype
        );
        FusedOperation(
            FUSED_OP_T type,
            const std::string& functor_name,
            const std::string& extra_code,
            const std::vector<FusedOperation>& arguments,
            DType dtype
        );
        // Convert through compilation to a usable Array operation than can be assigned
        operator Assignable<Array>() const;
        // Get the current dimensionality of the operation
        int ndim() const;
        // Checks whether dimensionality is zero (e.g. scalar, or op on a scalar)
        bool is_scalar() const;
        // Get the current return type of the operation
        DType dtype() const;
        // Get the current FusedOperation type
        const FUSED_OP_T& type() const;
        // Get the current broadcastable shape of the operation
        std::vector<int> bshape() const;
        // Get the current shape of the operation
        std::vector<int> shape() const;
        // Get the current size of the operation (if evaluated)
        int number_of_elements() const;
        // Get the array held by this operation
        const Array& array() const;
        // Get name of the associated functor
        const std::string& functor_name() const;
        // Get support code for this operation
        const std::string& extra_code() const;
        // Get the arguments of this functor
        const std::vector<FusedOperation>& arguments() const;
        // Find the lowest rank that allows computation of this operation
        int computation_rank() const;
        // Compile or load a callable function that runs the operation
        // parametrized by the operator used, the contiguity of the output,
        // the return type, and the device
        std::function<void(Array&, const std::vector<Array>&, const std::vector<double>&)> compile(
            const OPERATOR_T& operator_t,
            bool dst_contiguous,
            DType dtype,
            memory::Device device) const;
        static int type_to_min_rank(FUSED_OP_T type);
        static bool dtype_compatible(const FusedOperation& a, const FusedOperation& b);
        static bool ndim_compatible(const FusedOperation& a, const FusedOperation& b);
        // type_promotion
        // Find the most appropriate return type given two inputs.
        // If one argument is a scalar, then the type of the non-scalar
        // is used. If arguments are both scalars or both arrays, then
        // the most precise type is used:
        // - int & double -> double,
        // - int & float -> float,
        // - double & float -> double
        static DType type_promotion(const FusedOperation& a, const FusedOperation& b);
    private:
        std::vector<FusedOperation> arguments_;
        FUSED_OP_T type_;
        double scalar_;
        Array arr_;
        DType dtype_;
        std::string functor_name_;
        std::string extra_code_;

        // list of all Arrays used in this operation (can contain repeats)
        std::vector<Array> get_arrays(const std::vector<int>& bshape) const;
        // Append the arrays used by this operation to `arrays`
        void get_arrays(std::vector<Array>* arrays, const std::vector<int>& bshape) const;
        // list of all scalars used in this operation (can contain repeats)
        std::vector<double> get_scalars() const;
        // Append the scalars used by this operation to `scalars`
        void get_scalars(std::vector<double>* scalars) const;
        // generate the necessary constructors to wrap inside views the input arguments.
        std::string get_code_setup(memory::Device device, int rank) const;
        std::string get_code_setup(memory::Device device, int rank, int& arg_idx, int& scalar_arg_idx) const;
        // Return the calling code for accessing an element of a specific rank in an array_view
        std::string get_call_nd(int rank) const;
        // generate (recursively) the application of the current function
        std::string get_call_code_nd() const;
        std::string get_call_code_nd(int& arg_idx, int& scalar_arg_idx) const;

        // generate (recursively) the additional kernels or support code that each operation needs
        std::string get_extra_code() const;

        // generate (recursively) the application + assignment to output of the current function
        std::string get_assign_code_nd(const OPERATOR_T& operator_t, const std::string& call_nd) const;
        // generate the necessary code to compile this operation for a specific type, operator, rank,
        // and device
        std::string get_code_template(const OPERATOR_T&, bool, DType output_dtype, memory::Device, int) const;
};

namespace std {
    template<>
    struct hash<FusedOperation::FUSED_OP_T> {
        std::size_t operator()(const FusedOperation::FUSED_OP_T& k) const;
    };
}

namespace op2 {
    FusedOperation all_reduce(const FusedOperation& a,
                              const std::string& reducer_name,
                              DType return_type);
    FusedOperation all_reduce(const FusedOperation& a,
                              const std::string& reducer_name);
    // elementwise kernel given by name.
    // will assume that return type of kernel
    // is given by the `dtype` argument.
    FusedOperation elementwise(
        const FusedOperation& a,
        const std::string& functor_name,
        DType dtype
    );

    // elementwise kernel given by name. assumes
    // return type is unchanged from a's
    FusedOperation elementwise(
        const FusedOperation& a,
        const std::string& functor_name
    );

    // pair-wise kernel. Will type promote arguments
    // so that they have the same type when
    // given to the functor:
    // - float w/. double => double
    // - float w/. int => float
    // - double w/. int => double
    FusedOperation elementwise(
        const FusedOperation& a,
        const FusedOperation& b,
        const std::string& functor_name
    );

    // call a kernel on a pair of arguments. Assumes
    // both arguments should be of the same type. Peforms
    // type promotion on the arguments if not. Will paste
    // and run the associated code `kernel_code` during
    // compilation and usage. (Warning: this might cause
    // collisions when a name is used multiple times)
    FusedOperation binary_kernel_function(
        const FusedOperation& a,
        const FusedOperation& b,
        const std::string& function_name,
        const std::string& kernel_code
    );


    // Perform a type conversion by casting the values in x
    // to another dtype.
    FusedOperation astype(const FusedOperation& x, DType dtype);
} // namespace op2

#endif  // DALI_ARRAY_OP2_COMBINABLE_H
