#include "dali/array/function/args/reduce_over_args.h"
#include "dali/array/function/args/property_reducer.h"




template<typename Class, typename... Args>
BaseLazyFunction<Class,Args...>::BaseLazyFunction(Args... args) :
        bshape_(Class::lazy_output_bshape(args...)),
        dtype_(Class::lazy_output_dtype(args...)) {
}

template<typename Class, typename... Args>
std::vector<int> BaseLazyFunction<Class,Args...>::lazy_output_bshape(const Args&... args) {
    return ReduceOverArgs<BShapeCompatibleForAllArgsReducer>::reduce(args...);
}

template<typename Class, typename... Args>
DType BaseLazyFunction<Class,Args...>::lazy_output_dtype(const Args&... args) {
    return ReduceOverArgs<DTypeEqualForAllArgsReducer>::reduce(args...);
}

template<typename Class, typename... Args>
const std::vector<int>& BaseLazyFunction<Class,Args...>::bshape() const {
    return bshape_;
}

template<typename Class, typename... Args>
const DType& BaseLazyFunction<Class,Args...>::dtype() const {
    return dtype_;
}

template<typename Class, typename... Args>
const int BaseLazyFunction<Class,Args...>::evaluation_dim = 2;
