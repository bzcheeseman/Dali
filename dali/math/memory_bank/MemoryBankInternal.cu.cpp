#include "dali/math/memory_bank/MemoryBankInternal.h"

#include <mshadow/tensor.h>


#ifdef DALI_USE_CUDA
    template<typename R>
    mshadow::Tensor<mshadow::gpu, 2, R> dummy_gpu(R* ptr, int total_memory, int inner_dimension) {
        return mshadow::Tensor<mshadow::gpu, 2, R>(ptr, mshadow::Shape2(
                total_memory / inner_dimension, inner_dimension));
    }
#endif

template<typename R>
mshadow::Tensor<mshadow::cpu, 2, R> dummy_cpu(R* ptr, int total_memory, int inner_dimension) {
     return mshadow::Tensor<mshadow::cpu, 2, R>(ptr, mshadow::Shape2(
            total_memory / inner_dimension, inner_dimension));
}

template<typename R>
void memory_operations<R>::clear_cpu_memory(R* ptr, int amount, int inner_dimension) {
    auto dummy = dummy_cpu<R>(ptr, amount, inner_dimension);
    dummy = (R)0.0;
}


template<typename R>
R* memory_operations<R>::allocate_cpu_memory(int amount, int inner_dimension) {
    auto dummy = dummy_cpu<R>(NULL, amount, inner_dimension);
    mshadow::AllocSpace(&dummy, false);
    return dummy.dptr_;
}

template<typename R>
void memory_operations<R>::free_cpu_memory(R* addr, int amount, int inner_dimension) {
    auto dummy = dummy_cpu<R>(addr, amount, inner_dimension);
    mshadow::FreeSpace(&dummy);
}


template<typename R>
void memory_operations<R>::copy_memory_cpu_to_cpu(R* dest, R* source, int amount, int inner_dimension) {
    auto dummy_dest   = dummy_cpu<R>(dest,   amount, inner_dimension);
    auto dummy_source = dummy_cpu<R>(source, amount, inner_dimension);
    mshadow::Copy(dummy_dest, dummy_source);
}


#ifdef DALI_USE_CUDA
    template<typename R>
    size_t memory_operations<R>::cuda_available_memory() {
        size_t free_memory;
        size_t total_memory;
        cudaMemGetInfo(&free_memory, &total_memory);
        return free_memory;
    }
    template<typename R>
    void memory_operations<R>::clear_gpu_memory(R* ptr, int amount, int inner_dimension) {
        auto dummy = dummy_gpu<R>(ptr, amount, inner_dimension);
        dummy = (R)0.0;
    }

    template<typename R>
    R* memory_operations<R>::allocate_gpu_memory(int amount, int inner_dimension) {
        auto dummy = dummy_gpu<R>(NULL, amount, inner_dimension);
        mshadow::AllocSpace(&dummy, false);
        return dummy.dptr_;
    }

    template<typename R>
    void memory_operations<R>::free_gpu_memory(R* addr, int amount, int inner_dimension) {
        auto dummy = dummy_gpu<R>(addr, amount, inner_dimension);
        mshadow::FreeSpace(&dummy);
    }

    template<typename R>
    void memory_operations<R>::copy_memory_gpu_to_cpu(R* dest, R* source, int amount, int inner_dimension) {
        auto dummy_dest   = dummy_cpu<R>(dest,   amount, inner_dimension);
        auto dummy_source = dummy_gpu<R>(source, amount, inner_dimension);
        mshadow::Copy(dummy_dest, dummy_source);
    }

    template<typename R>
    void memory_operations<R>::copy_memory_gpu_to_gpu(R* dest, R* source, int amount, int inner_dimension) {
        auto dummy_dest   = dummy_gpu<R>(dest,   amount, inner_dimension);
        auto dummy_source = dummy_gpu<R>(source, amount, inner_dimension);
        mshadow::Copy(dummy_dest, dummy_source);
    }

    template<typename R>
    void memory_operations<R>::copy_memory_cpu_to_gpu(R* dest, R* source, int amount, int inner_dimension) {
        auto dummy_dest   = dummy_gpu<R>(dest,   amount, inner_dimension);
        auto dummy_source = dummy_cpu<R>(source, amount, inner_dimension);
        mshadow::Copy(dummy_dest, dummy_source);
    }
#endif

template class memory_operations<float>;
template class memory_operations<double>;
template class memory_operations<int>;
