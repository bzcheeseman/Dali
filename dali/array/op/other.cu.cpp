#include "dali/array/op/elementwise.h"

#include <iostream>

#include "dali/utils.h"
#include "dali/array/array.h"
#include "dali/array/array_function.h"
#include "dali/array/memory/device.h"
#include "dali/array/TensorFunctions.h"

using memory::Device;

struct IsNan : public Function<IsNan, bool, Array> {
    template<int devT, typename T>
    bool run(Array input, memory::Device dev) {}

    template<typename T>
    bool run(MArray<memory::DEVICE_T_CPU, T> input) {
        int num_elts = input.array.number_of_elements();
        return std::isnan(std::accumulate(input.ptr(), input.ptr() + num_elts, 0.0));
    }

#ifdef DALI_USE_CUDA
    template<typename T>
    bool run(MArray<memory::DEVICE_T_GPU, T> input) {
        int num_elts = input.number_of_elements();

        return std::is_nan(thrust::reduce(
            a.to_thrust(),
            a.to_thrust() + num_elts,
            0.0,
            thrust::plus<R>()
        ));
    }
#endif
};
