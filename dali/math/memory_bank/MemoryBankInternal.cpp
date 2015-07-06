#include "MemoryBankInternal.h"
using std::vector;

#ifdef DALI_USE_CUDA
template<typename R>
std::unordered_map<unsigned long long, vector<R *>>& gpu_memory_bank () {
    static std::unordered_map<unsigned long long, vector<R *>> bank;
    return bank;
};
template std::unordered_map<unsigned long long, vector<float *>>&  gpu_memory_bank ();
template std::unordered_map<unsigned long long, vector<double *>>& gpu_memory_bank ();
template std::unordered_map<unsigned long long, vector<int *>>&    gpu_memory_bank ();
#endif

template<typename R>
std::unordered_map<unsigned long long, vector<R *>>& cpu_memory_bank () {
    static std::unordered_map<unsigned long long, vector<R *>> bank;
    return bank;
};
template std::unordered_map<unsigned long long, vector<float *>>&  cpu_memory_bank ();
template std::unordered_map<unsigned long long, vector<double *>>& cpu_memory_bank ();
template std::unordered_map<unsigned long long, vector<int *>>&    cpu_memory_bank ();

