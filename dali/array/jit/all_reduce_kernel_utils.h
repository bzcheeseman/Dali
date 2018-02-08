#ifndef DALI_ARRAY_JIT_ALL_REDUCE_KERNEL_UTILS_H
#define DALI_ARRAY_JIT_ALL_REDUCE_KERNEL_UTILS_H
#include <string>
#include <functional>

void create_all_reduce_kernel_caller(int ndim, std::function<void(const std::string&)>);
void create_argument_all_reduce_kernel_caller(int ndim, std::function<void(const std::string&)>);
void create_axis_reduce_kernel_caller(int ndim, std::function<void(const std::string&)>);
void create_warp_axis_reduce_kernel_caller(int ndim, std::function<void(const std::string&)>);
void create_warp_all_reduce_kernel_caller(int ndim, std::function<void(const std::string&)>);
void create_argument_axis_reduce_kernel_caller(int ndim, std::function<void(const std::string&)>);

#endif  // DALI_ARRAY_JIT_ALL_REDUCE_KERNEL_UTILS_H
