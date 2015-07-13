#include "dali/math/memory_bank/MemoryBank.h"
using std::vector;

static std::mutex memory_mutex;

template<typename R>
std::unordered_map<unsigned long long,std::vector<R*>> memory_bank<R>::cpu_memory_bank;


template<typename R>
void memory_bank<R>::deposit_cpu(int amount, int inner_dimension, R* ptr) {
    // make sure there is only one person
    // at a time in the vault to prevent
    // robberies
    std::lock_guard<decltype(memory_mutex)> guard(memory_mutex);
    cpu_memory_bank[amount].emplace_back(ptr);
}

template<typename R>
R* memory_bank<R>::allocate_cpu(int amount, int inner_dimension) {
    {
        std::lock_guard<decltype(memory_mutex)> guard(memory_mutex);
        if (cpu_memory_bank.find(amount) != cpu_memory_bank.end()) {
            auto& deposit_box = cpu_memory_bank.at(amount);
            if (deposit_box.size() > 0) {
                auto preallocated_memory = deposit_box.back();
                deposit_box.pop_back();
                return preallocated_memory;
            }
        }
    }
    num_cpu_allocations++;
    total_cpu_memory+= amount;
    return memory_operations<R>::allocate_cpu_memory(amount, inner_dimension);
}

template<typename R>
std::atomic<long long> memory_bank<R>::num_cpu_allocations(0);

template<typename R>
std::atomic<long long> memory_bank<R>::total_cpu_memory(0);

#ifdef DALI_USE_CUDA


    template<typename R>
    std::unordered_map<unsigned long long,std::vector<R*>> memory_bank<R>::gpu_memory_bank;

    template<typename R>
    std::atomic<long long> memory_bank<R>::num_gpu_allocations(0);

    template<typename R>
    std::atomic<long long> memory_bank<R>::total_gpu_memory(0);

    template<typename R>
    void memory_bank<R>::deposit_gpu(int amount, int inner_dimension, R* ptr) {
        std::lock_guard<decltype(memory_mutex)> guard(memory_mutex);
        gpu_memory_bank[amount].emplace_back(ptr);
    }

    template<typename R>
    R* memory_bank<R>::allocate_gpu(int amount, int inner_dimension) {
        std::lock_guard<decltype(memory_mutex)> guard(memory_mutex);
        if (gpu_memory_bank.find(amount) != gpu_memory_bank.end()) {
            auto& deposit_box = gpu_memory_bank.at(amount);
            if (deposit_box.size() > 0) {
                auto preallocated_memory = deposit_box.back();
                deposit_box.pop_back();
                return preallocated_memory;
            }
        }

        num_gpu_allocations++;
        total_gpu_memory+= amount;
        return memory_operations<R>::allocate_gpu_memory(amount, inner_dimension);
    }
#endif

template class memory_bank<float>;
template class memory_bank<double>;
template class memory_bank<int>;
