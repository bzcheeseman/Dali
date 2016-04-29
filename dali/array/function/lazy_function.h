#ifndef DALI_ARRAY_FUNCTION_LAZY_FUNCTION_H
#define DALI_ARRAY_FUNCTION_LAZY_FUNCTION_H

#include <vector>

#include "dali/array/dtype.h"
#include "dali/array/function/lazy_evaluator.h"
#include "dali/array/function/args/reduce_over_args.h"
#include "dali/array/function/args/property_reducer.h"

template<typename Class, typename... Args>
struct LazyFunction : public LazyExp<Class> {
    std::vector<int> shape_;
    DType dtype_;

    LazyFunction(Args... args) :
            shape_(Class::lazy_output_shape(args...)),
            dtype_(Class::lazy_output_dtype(args...)) {
    }

    static std::vector<int> lazy_output_shape(Args... args) {
        return ReduceOverArgs<ShapeEqualForAllArgsReducer>::reduce(args...);
    }

    static DType lazy_output_dtype(Args... args) {
        return ReduceOverArgs<DTypeEqualForAllArgsReducer>::reduce(args...);
    }

    const std::vector<int>& shape() const {
        return shape_;
    }

    const DType& dtype() const {
        return dtype_;
    }

    AssignableArray as_assignable() const {
        return LazyEvaluator<Class>::run(this->self());
    }
};

#endif
