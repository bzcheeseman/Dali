#include "elementwise.h"

#include <iostream>

#include "dali/utils.h"
#include "dali/array/array.h"
#include "dali/array/array_function.h"

struct Sigmoid : public Function<Sigmoid, Array, Array> {

    template<int dev, typename T>
    Array operator()(TypedArray<dev,T> a) {
        std::cout << "performing squaring : ("
        	      << memory_ops::device_to_name[dev] << ", "
        	      << type_to_name<T>()
        	      << ")"
			      << std::endl;
        return Array();
    }

    template<int dev>
    Array operator()(TypedArray<dev,int> a) {
        throw std::string("ERROR: We did not implement sigmoid for ints, suck it");
        return Array();
    }

    FAIL_ON_OTHER_CASES(Sigmoid);
};

Array sigmoid(const Array& x) {
    ELOG("sigmoid sigmoid sigmoid");
	return Sigmoid::eval(x);
}
