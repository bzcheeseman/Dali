#ifndef DALI_ARRAY_OP2_ALL_REDUCE_KERNEL_UTILS_H
#define DALI_ARRAY_OP2_ALL_REDUCE_KERNEL_UTILS_H
#include <string>

std::string create_all_reduce_kernel_caller(int ndim, int result_ndim);
std::string create_argument_all_reduce_kernel_caller(int ndim, int result_ndim);

#endif  // DALI_ARRAY_OP2_ALL_REDUCE_KERNEL_UTILS_H
