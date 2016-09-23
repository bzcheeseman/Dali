#ifndef DALI_ARRAY_OP2_RTC_UTILS_H
#define DALI_ARRAY_OP2_RTC_UTILS_H
#include <string>
#include <vector>
#include "dali/array/array.h"
#include "dali/array/dtype.h"
#include "dali/array/memory/device.h"

// Create declaration code for wrapping a variable inside a new view
std::string build_view_constructor(
    const std::string& cpp_type, bool contiguous, int rank, const std::string& varname
);
// create declaration for copying a scalar argument into a local variable
std::string build_scalar_constructor(
	const std::string& cpp_type, int rank, int start_arg
);
// Create declaration code for wrapping several variables
// contained in a vector named `arguments` and naming the views `arg_[i]_view` for
// each index i in `arguments`.
std::string build_views_constructor(
    const std::string& cpp_type,
    const std::vector<bool>& contiguous,
    int rank,
    int start_arg);

// Declare a nested c++ for loop for a specific rank (dimensionality)
// that calls `code` in the center of the loop (modified a variable
// of type `Shape` named "query" contains the current indices for the loop)
std::string construct_for_loop(int rank, const std::string& code, const std::string& varname, int indent);

// If the array `out` is uninitialized, then give it a specific type
// device, and shape (filled with zeros), else check that it is
// compatible with the desired dimensions and type.
void initialize_output_array(Array& out,
                             const DType& output_dtype,
                             const memory::Device& output_device,
                             std::vector<int>* output_bshape_ptr);

// Find the shape that contains all the other shapes in the vector `bshapes`
// while maintaining broadcasted dimensions where possible, and checking for
// mismatches elsewhere. Scalar arguments are ignored when computing the
// output shape.
std::vector<int> get_function_bshape(const std::vector<std::vector<int>>& bshapes);

#endif
