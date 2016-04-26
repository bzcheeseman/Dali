#ifndef DALI_ARRAY_THRUST_UTILS_H
#define DALI_ARRAY_THRUST_UTILS_H

#include "dali/config.h"

#ifdef DALI_USE_CUDA
    #include <thrust/device_vector.h>
    #include <thrust/equal.h>
    #include <thrust/functional.h>
    #include <thrust/reduce.h>
    #include <thrust/transform.h>
    #include <thrust/random.h>
    #include <thrust/sort.h>
    #include <thrust/sequence.h>
    #include <thrust/transform_reduce.h>
    // contains thrust::max_element & thrust::min_element
    #include <thrust/extrema.h>

    #define STR(x) __THIS_IS_VERY_ABNOXIOUS(x)
    #define __THIS_IS_VERY_ABNOXIOUS(tok) #tok

    /* CUDA UTILS START HERE */
    namespace tensor_ops {
        template<typename R, int ndims>
        thrust::device_ptr<R> to_thrust(const mshadow::Tensor<mshadow::gpu, ndims, R>& tg) {
            auto dev_ptr = thrust::device_pointer_cast(tg.dptr_);
            return dev_ptr;
        }
    }
#endif

#endif
