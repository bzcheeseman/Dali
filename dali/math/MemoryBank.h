#ifndef DALI_MATH_MEMORY_BANK_H
#define DALI_MATH_MEMORY_BANK_H
#include <unordered_map>
#include <vector>
/*
Memory Bank
-----------

Here Dali stores the treasures of its
many conquests battling memory allocations
and gpu monsters.

The key is the memory buffer size desired, and
the value is the list of available buffers.
*/
#ifdef DALI_USE_CUDA
template<typename R>
std::unordered_map<unsigned long long, std::vector<R *>>& gpu_memory_bank ();
#endif

template<typename R>
std::unordered_map<unsigned long long, std::vector<R *>>& cpu_memory_bank ();

#endif
