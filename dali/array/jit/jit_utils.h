#ifndef DALI_ARRAY_JIT_JIT_UTILS_H
#define DALI_ARRAY_JIT_JIT_UTILS_H
#include <string>
#include <vector>
#include "dali/array/array.h"
#include "dali/array/dtype.h"
#include "dali/array/memory/device.h"
#include "dali/array/jit/jit_runner.h"

// Create declaration code for wrapping a variable inside a new view
std::string build_array_definition(const std::string& cpp_type,
                                   const std::string& varname,
                                   bool contiguous,
                                   int rank,
                                   const std::string& constructor_arguments);

// create declaration for copying a scalar argument into a local variable
std::string build_scalar_definition(const std::string& cpp_type,
                                    const std::string& varname,
                                    int rank,
                                    const std::string& captured_name);

std::string build_shape_definition(const std::string& varname,
                                   int rank,
								                   const std::string& captured_name);


// Given the number of indexes (rank) returns string to access kernels.
std::string generate_accessor_string(int rank);


// Declare a nested c++ for loop for a specific rank (dimensionality)
// that calls `code` in the center of the loop (modified a variable
// of type `Shape` named "query" contains the current indices for the loop)
std::string construct_for_loop(int rank, std::string code, const std::string& varname, int indent);

// check that output Array matches the desired dtype and shape.
void ensure_output_array_compatible(const Array& out,
                                    const DType& output_dtype,
                                    const std::vector<int>& output_shape);

// Find the shape that contains all the other shapes in the vector `shapes`.
// Scalar arguments are ignored when computing the output shape.
std::vector<int> get_common_shape(const std::vector<const std::vector<int>*>& shapes);
std::vector<int> get_common_shape(const std::vector<Array>& arrays);

std::string define_kernel(int ndim, bool has_shape,
                          const std::vector<std::string>& arguments,
                          std::string kernel, std::string kernel_name,
                          bool is_assignable);
std::string generate_call_code_nd(const Expression*,
                                  const std::string& kernel_name,
                                  const op::jit::SymbolTable& symbol_table,
                                  const op::jit::node_to_info_t& node_to_info,
                                  memory::DeviceT device_type,
                                  bool has_shape);
#endif
