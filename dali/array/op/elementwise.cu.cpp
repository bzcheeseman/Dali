#include "dali/array/op/elementwise.h"

#include <iostream>

#include "dali/utils.h"
#include "dali/array/array.h"
#include "dali/array/array_function.h"
#include "dali/array/memory/device.h"
#include "dali/array/TensorFunctions.h"

using memory::Device;


struct Sigmoid : public Function<Sigmoid, Array, Array> {

    template<int devT, typename T>
    Array run(Array input, Device dev) {
        Array out(input.shape(), input.dtype());

        auto m = getmshadow<devT,T>{dev};

        m.d1(out) = mshadow::expr::F<TensorOps::op::sigmoid<T>>(m.d1(input));
        return out;

    }

    FAIL_ON_OTHER_CASES(Sigmoid);
};

Array sigmoid(const Array& x) {
    return Sigmoid::eval(x);
}
