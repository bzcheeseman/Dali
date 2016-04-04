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

        auto m_in = getmshadow<devT,T>::oned(input, dev);
        auto m_out = getmshadow<devT,T>::oned(out, dev);


        m_out = mshadow::expr::F<TensorOps::op::sigmoid<T>>(m_in);
        return out;

    }

    FAIL_ON_OTHER_CASES(Sigmoid);
};

Array sigmoid(const Array& x) {
    return Sigmoid::eval(x);
}
