#include "other.h"

#include <cmath>
#include <iostream>

#include "dali/config.h"
#include "dali/utils.h"
#include "dali/array/array.h"
#include "dali/array/function/function.h"
#include "dali/array/memory/device.h"
#include "dali/array/lazy/binary.h"
#include "dali/array/lazy/reducers.h"
#include "dali/array/lazy/unary.h"

using memory::Device;

struct IsNan : public NonArrayFunction<IsNan, bool, Array> {
    template<OPERATOR_T operator_t, typename T>
    void typed_eval(bool* out, TypedArray<memory::DEVICE_T_CPU, T> input) {
        ASSERT2(input.array.spans_entire_memory(), "At this time is_nan is not available for views");
        int num_elts = input.array.number_of_elements();
        *out = std::isnan(std::accumulate(input.ptr(), input.ptr() + num_elts, 0.0));
    }

#ifdef DALI_USE_CUDA
    template<OPERATOR_T operator_t, typename T>
    void typed_eval(bool* out, TypedArray<memory::DEVICE_T_GPU, T> input) {
        ASSERT2(input.array.contiguous_memory(), "At this time is_nan is not available for views");

        int num_elts = input.array.number_of_elements();
        // TODO(szymon, jonathan): switch to mshadow sum all (to avoid thrust overhead)
        *out = std::isnan(thrust::reduce(
            input.to_thrust(),
            input.to_thrust() + num_elts,
            0.0,
            thrust::plus<T>()
        ));
    }
#endif
};

namespace op {
    bool is_nan(const Array& x) { return IsNan::run(x); }
    Assignable<Array> all_equals(const Array& left, const Array& right) {
        return lazy::product(lazy::equals(left, right));
    }

    Assignable<Array> all_close(const Array& left, const Array& right, const double& atolerance) {
        ASSERT2(atolerance >= 0,
            utils::MS() << "atolerance must be a strictly positive number (got atolerance="
                        << atolerance << ").");
        return lazy::product(
            lazy::lessthanequal(
                lazy::abs(
                    lazy::sub(left,right)
                ),
                atolerance
            )
        );
    }
} // namespace op

