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

//TODO(jonathan,szymon): add equality tests here (abs difference, and exact equals)

template<typename FillT>
struct Fill : public Function<Fill<FillT>, Array, FillT> {
    static const bool disable_output_shape_check = true;
    static const bool disable_output_dtype_check = true;

    static DType deduce_output_dtype(const FillT& filler) {
        return template_to_dtype<FillT>();
    }

    static std::vector<int> deduce_output_shape(const FillT& filler) {
        return {};
    }

    static memory::Device deduce_output_device(const FillT&) {
        return memory::default_preferred_device;
    }

    template<int devT, typename T>
    void typed_eval(MArray<devT, T> input, const FillT& filler) {
        input.d1(memory::AM_OVERWRITE) = (T) filler;
    }
};


template<typename T>
AssignableArray fill(const T& other) {
    return Fill<T>::run(other);
}

template AssignableArray fill(const float&);
template AssignableArray fill(const double&);
template AssignableArray fill(const int&);

bool is_nan(const Array& x) { return IsNan::run(x); }
