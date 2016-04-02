#ifndef DALI_ARRAY_THRUSTALLOCATOR_H
#define DALI_ARRAY_THRUSTALLOCATOR_H

#include "dali/config.h"

#ifdef DALI_USE_CUDA
#include <thrust/device_vector.h>
#include <thrust/device_malloc_allocator.h>

// inspiration from
// https://parallel-computing.pro/index.php/9-cuda/34-thrust-cuda-tip-reuse-temporary-buffers-across-transforms
template<typename R>
struct cached_allocator : thrust::device_malloc_allocator<R>{
    typedef thrust::device_malloc_allocator<R> super_t;
    typedef typename super_t::pointer   pointer;
    typedef typename super_t::size_type size_type;
    cached_allocator();
    ~cached_allocator();
    pointer allocate(size_type num_bytes);
    void deallocate(pointer p, size_type n);
};

#endif
#endif
