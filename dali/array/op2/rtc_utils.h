#ifndef DALI_ARRAY_OP2_RTC_UTILS_H
#define DALI_ARRAY_OP2_RTC_UTILS_H
#include <string>
#include <vector>
#include "dali/array/array.h"
#include "dali/array/dtype.h"
#include "dali/array/memory/device.h"

// Create declaration code for wrapping a variable inside a new view
std::string build_array_definition(const std::string& cpp_type,
                                   const std::string& varname,
                                   bool contiguous,
                                   int rank,
                                   const std::string& captured_name);

// create declaration for copying a scalar argument into a local variable
std::string build_scalar_definition(const std::string& cpp_type,
                                     const std::string& varname,
                                     int rank,
                                     const std::string& captured_name);


// Given the number of indexes (rank) returns string to access kernels.
std::string generate_accessor_string(int rank);


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
std::vector<int> get_common_bshape(const std::vector<std::vector<int>>& bshapes);

#endif
