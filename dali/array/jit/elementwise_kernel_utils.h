#ifndef DALI_ARRAY_JIT_ELEMENTWISE_KERNEL_UTILS_H
#define DALI_ARRAY_JIT_ELEMENTWISE_KERNEL_UTILS_H
#include <string>
#include <functional>

std::string elementwise_kernel_name(int num_args, int ndim);
void create_elementwise_kernel_caller(int num_args, int ndim,
	std::function<void(const std::string&)>);

#endif  // DALI_ARRAY_JIT_ELEMENTWISE_KERNEL_UTILS_H
