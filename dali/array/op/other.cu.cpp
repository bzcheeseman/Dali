#include "dali/array/op/elementwise.h"

#include <cmath>
#include <iostream>

#include "dali/config.h"
#include "dali/utils.h"
#include "dali/array/array.h"
#include "dali/array/function/function.h"
#include "dali/array/memory/device.h"
#include "dali/array/TensorFunctions.h"

using memory::Device;

struct IsNan : public NonArrayFunction<IsNan, bool, Array> {
    template<typename T>
    void typed_eval(bool* out, MArray<memory::DEVICE_T_CPU, T> input) {
        int num_elts = input.array.number_of_elements();
        *out = std::isnan(std::accumulate(input.ptr(), input.ptr() + num_elts, 0.0));
    }

#ifdef DALI_USE_CUDA
    template<typename T>
    void typed_eval(bool* out, MArray<memory::DEVICE_T_GPU, T> input) {
        int num_elts = input.array.number_of_elements();

        *out = std::isnan(thrust::reduce(
            input.to_thrust(),
            input.to_thrust() + num_elts,
            0.0,
            thrust::plus<T>()
        ));
    }
#endif
};

bool is_nan(const Array& x) { return IsNan::run(x); }
