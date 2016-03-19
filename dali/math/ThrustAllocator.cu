#include "dali/math/ThrustAllocator.h"

#include "dali/math/memory_bank/MemoryBank.h"

#ifdef DALI_USE_CUDA
    template<typename R>
    cached_allocator<R>::cached_allocator() {}

    template<typename R>
    cached_allocator<R>::~cached_allocator() {}

    template<typename R>
    typename cached_allocator<R>::pointer cached_allocator<R>::allocate(size_type num_bytes) {
        auto ptr = memory_bank<float>::allocate_gpu(
            num_bytes * sizeof(R) / sizeof(float),
            num_bytes * sizeof(R) / sizeof(float)
        );
        return thrust::device_pointer_cast((R*)ptr);
    }

    template<typename R>
    void cached_allocator<R>::deallocate(pointer ptr, size_type n) {
        memory_bank<float>::deposit_gpu(
            n * sizeof(R) / sizeof(float),
            n * sizeof(R) / sizeof(float),
            (float*)thrust::raw_pointer_cast(ptr)
        );
    }

    template class cached_allocator<float>;
    template class cached_allocator<double>;
    template class cached_allocator<int>;
    template class cached_allocator<char>;
    template class cached_allocator<uint>;
#endif
