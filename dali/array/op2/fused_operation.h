#ifndef DALI_ARRAY_OP2_COMBINABLE_H
#define DALI_ARRAY_OP2_COMBINABLE_H

#include <vector>
#include "dali/array/array.h"

class FusedOperation {
    public:
        // A fused operation taking only an array is no-op
        FusedOperation(const Array& arr);
        FusedOperation(Array&& arr);
        FusedOperation(const Assignable<Array>& arr);
        // A fused operation can be constructed using a type identifier
        // 0 -> no-op holding an Array
        // 1 -> elementwise pair kernel
        // 2 -> binary_kernel
        FusedOperation(
            int type,
            const std::string& functor_name,
            const std::vector<FusedOperation>& arguments
        );
        FusedOperation(
            int type,
            const std::string& functor_name,
            const std::string& extra_code,
            const std::vector<FusedOperation>& arguments
        );
        // Convert through compilation to a usable Array operation than can be assigned
        operator Assignable<Array>() const;
        // Get the current dimensionality of the operation
        int ndim() const;
        // Get the current return type of the operation
        DType dtype() const;
        // Get the current FusedOperation type
        const int& type() const;
        // Get the current broadcastable shape of the operation
        std::vector<int> bshape() const;
        // Get the array held by this operation
        const Array& array() const;
        // Get name of the associated functor
        const std::string& functor_name() const;
        // Get the arguments of this functor
        const std::vector<FusedOperation>& arguments() const;
        // Find the lowest rank that allows computation of this operation
        int computation_rank() const;
        // Compile or load a callable function that runs the operation
        // parametrized by the operator used, the contiguity of the output,
        // the return type, and the device
        std::function<void(Array&, const std::vector<Array>&)> compile(
            const OPERATOR_T& operator_t,
            bool dst_contiguous,
            DType dtype,
            memory::Device device) const;
        static int type_to_min_rank(int type);
    private:
        std::vector<FusedOperation> arguments_;
        int type_;
        Array arr_;
        std::string functor_name_;
        std::string extra_code_;

        // list of all Arrays used in this operation (can contain repeats)
        std::vector<Array> get_arrays() const;
        // Append the arrays used by this operation to `arrays`
        void get_arrays(std::vector<Array>* arrays) const;
        // generate the necessary constructors to wrap inside views the input arguments.
        std::string get_code_setup(const std::string& cpp_type, memory::Device device, int rank, int& arg_idx) const;
        std::string get_code_setup(const std::string& cpp_type, memory::Device device, int rank) const;
        // Return the calling code for accessing an element of a specific rank in an array_view
        std::string get_call_nd(int rank) const;
        // generate (recursively) the application of the current function
        std::string get_call_code_nd(const std::string& cpp_type, const std::string& call_nd) const;
        std::string get_call_code_nd(const std::string& cpp_type, const std::string& call_nd, int& start_arg) const;

        // generate (recursively) the additional kernels or support code that each operation needs
        std::string get_extra_code() const;
        void get_extra_code(std::string* extra_code_ptr) const;

        // generate (recursively) the application + assignment to output of the current function
        std::string get_assign_code_nd(const OPERATOR_T&, const std::string&, const std::string&) const;
        // generate the necessary code to compile this operation for a specific type, operator, rank,
        // and device
        std::string get_code_template(const OPERATOR_T&, bool, DType dtype, memory::Device, int) const;
};

namespace op2 {
    FusedOperation elementwise(
        const FusedOperation& a,
        const std::string& functor_name
    );

    FusedOperation elementwise(
        const FusedOperation& a,
        const FusedOperation& b,
        const std::string& functor_name
    );

    FusedOperation binary_kernel_function(
        const FusedOperation& a,
        const FusedOperation& b,
        const std::string& function_name,
        const std::string& kernel_code
    );
} // namespace op2

#endif  // DALI_ARRAY_OP2_COMBINABLE_H
