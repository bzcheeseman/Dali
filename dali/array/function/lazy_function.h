#ifndef DALI_ARRAY_FUNCTION_LAZY_FUNCTION_H
#define DALI_ARRAY_FUNCTION_LAZY_FUNCTION_H

#include <vector>

#include "dali/array/dtype.h"
#include "dali/array/function/lazy_evaluator.h"
#include "dali/array/function/args/reduce_over_args.h"
#include "dali/array/function/args/property_reducer.h"

template<typename Class, typename... Args>
struct LazyFunction : public LazyExp<Class> {
    static const int evaluation_dim;
    std::vector<int> bshape_;
    DType dtype_;

    LazyFunction(Args... args) :
            bshape_(Class::lazy_output_bshape(args...)),
            dtype_(Class::lazy_output_dtype(args...)) {
    }

    static std::vector<int> lazy_output_bshape(const Args&... args) {
        return ReduceOverArgs<BShapeCompatibleForAllArgsReducer>::reduce(args...);
    }

    static DType lazy_output_dtype(const Args&... args) {
        return ReduceOverArgs<DTypeEqualForAllArgsReducer>::reduce(args...);
    }

    const std::vector<int>& bshape() const {
        return bshape_;
    }

    const DType& dtype() const {
        return dtype_;
    }

    AssignableArray as_assignable() const {
        return LazyEvaluator<Class>::run(this->self());
    }
};

template<typename Class, typename... Args>
const int LazyFunction<Class,Args...>::evaluation_dim = 2;


#endif
