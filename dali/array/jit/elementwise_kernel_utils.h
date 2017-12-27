#ifndef DALI_ARRAY_JIT_ELEMENTWISE_KERNEL_UTILS_H
#define DALI_ARRAY_JIT_ELEMENTWISE_KERNEL_UTILS_H
#include <string>

std::string elementwise_kernel_name(int num_args, int ndim);
std::string create_elementwise_kernel_caller(int num_args, int ndim);

#endif  // DALI_ARRAY_JIT_ELEMENTWISE_KERNEL_UTILS_H
