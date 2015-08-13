#include "dali/math/memory_bank/MemoryBank.h"
using std::vector;

static std::mutex memory_mutex;

template<typename R>
cuckoohash_map<unsigned long long,std::vector<R*>> memory_bank<R>::cpu_memory_bank(100000);


template<typename R>
void memory_bank<R>::deposit_cpu(int amount, int inner_dimension, R* ptr) {
    // make sure there is only one person
    // at a time in the vault to prevent
    // robberies
    /*cpu_memory_bank.upsert(amount, [ptr](std::vector<R*>& deposit_box) {
        deposit_box.emplace_back(ptr);
    }, {ptr});*/
    memory_operations<R>::free_cpu_memory(ptr, amount, inner_dimension);
}

template<typename R>
R* memory_bank<R>::allocate_cpu(int amount, int inner_dimension) {
    /*R* memory = NULL;
    bool success = cpu_memory_bank.update_fn(amount, [&memory](std::vector<R*>& deposit_box) {
        if (!deposit_box.empty()) {
            memory = deposit_box.back();
            deposit_box.pop_back();
        }
    });
    if (memory != NULL) {
        return memory;
    }*/
    num_cpu_allocations++;
    total_cpu_memory += amount;
    return memory_operations<R>::allocate_cpu_memory(amount, inner_dimension);
}

template<typename R>
void memory_bank<R>::clear_cpu() {
    for (auto it = cpu_memory_bank.cbegin(); !it.is_end(); it++) {
        auto& ptrs = it->second;
        for (auto ptr : ptrs) {
            memory_operations<R>::free_cpu_memory(ptr, it->first, 1);
        }
        total_cpu_memory -= (it->first) * ptrs.size();
    }
    cpu_memory_bank.clear();
}

template<typename R>
std::atomic<long long> memory_bank<R>::num_cpu_allocations(0);

template<typename R>
std::atomic<long long> memory_bank<R>::total_cpu_memory(0);

#ifdef DALI_USE_CUDA
    template<typename R>
    void memory_bank<R>::clear_gpu() {
        for (auto it = gpu_memory_bank.cbegin(); !it.is_end(); it++) {
            auto& ptrs = it->second;
            for (auto ptr : ptrs) {
                memory_operations<R>::free_gpu_memory(ptr, it->first, 1);
            }
            total_gpu_memory -= (it->first) * ptrs.size();
        }
        gpu_memory_bank.clear();
    }

    template<typename R>
    cuckoohash_map<unsigned long long,std::vector<R*>> memory_bank<R>::gpu_memory_bank(100000);

    template<typename R>
    std::atomic<long long> memory_bank<R>::num_gpu_allocations(0);

    template<typename R>
    std::atomic<long long> memory_bank<R>::total_gpu_memory(0);

    template<typename R>
    void memory_bank<R>::deposit_gpu(int amount, int inner_dimension, R* ptr) {
        gpu_memory_bank.upsert(amount, [ptr](std::vector<R*>& deposit_box) {
            deposit_box.emplace_back(ptr);
        }, {ptr});
    }

    template<typename R>
    R* memory_bank<R>::allocate_gpu(int amount, int inner_dimension) {
        R* memory = NULL;
        bool success = gpu_memory_bank.update_fn(amount, [&memory](std::vector<R*>& deposit_box) {
            if (!deposit_box.empty()) {
                memory = deposit_box.back();
                deposit_box.pop_back();
            }
        });
        if (memory != NULL) {
            return memory;
        }
        num_gpu_allocations++;
        total_gpu_memory += amount;
        return memory_operations<R>::allocate_gpu_memory(amount, inner_dimension);
    }
#endif

template class memory_bank<float>;
template class memory_bank<double>;
template class memory_bank<int>;
