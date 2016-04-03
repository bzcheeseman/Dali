#include "elementwise.h"

#include <iostream>

#include "dali/utils.h"
#include "dali/array/array.h"
#include "dali/array/array_function.h"

struct Sigmoid : public Function<Sigmoid, Array, Array> {

    template<int dev, typename T>
    Array operator()(TypedArray<dev,T> input) {

        // start by creating an empty container.
        auto out = TypedArray<dev,T>::empty_like(input);
        // then you apply some elementwise function to it
        // (before doing so, you must convert to an mshadow::expression)
        // out.mshadow_tensor() = F<op::sigmoid<R>>(input.mshadow_tensor());

        // std::cout << "performing squaring : ("
        //           << memory_ops::device_to_name[dev] << ", "
        //           << type_to_name<T>()
        //           << ")"
        //           << std::endl;

        // then return it
        return Array(std::move(out));
    }

    template<int dev>
    Array operator()(TypedArray<dev,int> a) {
        throw std::string("ERROR: sigmoid not implemented for int");
        return Array();
    }

    FAIL_ON_OTHER_CASES(Sigmoid);
};

Array sigmoid(const Array& x) {
    ELOG("sigmoid sigmoid sigmoid");
    return Sigmoid::eval(x);
}
