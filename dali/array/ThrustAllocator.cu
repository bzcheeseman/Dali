#include "dali/array/ThrustAllocator.h"

#include "dali/array/memory/memory_ops.h"
#include "dali/array/memory/memory_bank.h"

#ifdef DALI_USE_CUDA
    template<typename R>
    cached_allocator<R>::cached_allocator() {}

    template<typename R>
    cached_allocator<R>::~cached_allocator() {}

    template<typename R>
    typename cached_allocator<R>::pointer cached_allocator<R>::allocate(size_type num_bytes) {
        auto device_ptr = memory_bank::allocate(
            memory_ops::DEVICE_GPU,
            num_bytes * sizeof(R),
            num_bytes * sizeof(R)
        );
        return thrust::device_pointer_cast((R*)device_ptr.ptr);
    }

    template<typename R>
    void cached_allocator<R>::deallocate(pointer ptr, size_type n) {
        memory_bank::deposit(
            memory_ops::DevicePtr(memory_ops::DEVICE_GPU, thrust::raw_pointer_cast(ptr)),
            n * sizeof(R),
            n * sizeof(R)
        );
    }

    template class cached_allocator<float>;
    template class cached_allocator<double>;
    template class cached_allocator<int>;
    template class cached_allocator<char>;
    template class cached_allocator<uint>;
#endif
