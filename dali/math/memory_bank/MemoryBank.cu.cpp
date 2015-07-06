#include "dali/math/memory_bank/MemoryBank.h"
using std::vector;

static std::mutex memory_mutex;

template<typename R>
void memory_bank<R>::deposit_cpu(int amount, int inner_dimension, R* ptr) {
    // make sure there is only one person
    // at a time in the vault to prevent
    // robberies
    std::lock_guard<decltype(memory_mutex)> guard(memory_mutex);
    auto& bank = cpu_memory_bank<R>();
    bank[amount].emplace_back(ptr);
}

template<typename R>
R* memory_bank<R>::allocate_cpu(int amount, int inner_dimension) {
    {
        std::lock_guard<decltype(memory_mutex)> guard(memory_mutex);
        auto& bank = cpu_memory_bank<R>();
        if (bank.find(amount) != bank.end()) {
            auto& deposit_box = bank.at(amount);
            if (deposit_box.size() > 0) {
                auto preallocated_memory = deposit_box.back();
                deposit_box.pop_back();
                return preallocated_memory;
            }
        }
    }
    // create a dummy tensor to allocate memory
    // with correct pitch.
    mshadow::Tensor<mshadow::cpu, 2, R> dummy_tensor(
        mshadow::Shape2(
            amount / inner_dimension,
            inner_dimension
        )
    );
    mshadow::AllocSpace(&dummy_tensor, false);
    num_cpu_allocations++;
    total_cpu_memory+= amount;
    return dummy_tensor.dptr_;
}

template<typename R>
std::atomic<long long> memory_bank<R>::num_cpu_allocations(0);

template<typename R>
std::atomic<long long> memory_bank<R>::total_cpu_memory(0);

#ifdef DALI_USE_CUDA
    template<typename R>
    std::atomic<long long> memory_bank<R>::num_gpu_allocations(0);

    template<typename R>
    std::atomic<long long> memory_bank<R>::total_gpu_memory(0);

    template<typename R>
    void memory_bank<R>::deposit_gpu(int amount, int inner_dimension, R* ptr) {
        std::lock_guard<decltype(memory_mutex)> guard(memory_mutex);
        auto& bank = gpu_memory_bank<R>();
        bank[amount].emplace_back(ptr);
    }

    template<typename R>
    size_t memory_bank<R>::cuda_available_memory() {
        size_t free_memory;
        size_t total_memory;
        cudaMemGetInfo(&free_memory, &total_memory);
        return free_memory;
    }

    template<typename R>
    R* memory_bank<R>::allocate_gpu(int amount, int inner_dimension) {
        std::lock_guard<decltype(memory_mutex)> guard(memory_mutex);
        auto& bank = gpu_memory_bank<R>();
        if (bank.find(amount) != bank.end()) {
            auto& deposit_box = bank.at(amount);
            if (deposit_box.size() > 0) {
                auto preallocated_memory = deposit_box.back();
                deposit_box.pop_back();
                return preallocated_memory;
            }
        }
        // create a dummy tensor to allocate memory
        // with correct pitch.
        mshadow::Tensor<mshadow::gpu, 2, R> dummy_tensor(
            mshadow::Shape2(
                amount / inner_dimension,
                inner_dimension
            )
        );
        mshadow::AllocSpace(&dummy_tensor, false);
        num_gpu_allocations++;
        total_gpu_memory+= amount;
        return dummy_tensor.dptr_;
    }
#endif

template class memory_bank<float>;
template class memory_bank<double>;
template class memory_bank<int>;
