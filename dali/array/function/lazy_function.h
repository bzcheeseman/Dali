#ifndef DALI_ARRAY_FUNCTION_LAZY_FUNCTION_H
#define DALI_ARRAY_FUNCTION_LAZY_FUNCTION_H

#include <vector>

#include "dali/array/array.h"
#include "dali/array/dtype.h"
#include "dali/array/function/expression.h"

namespace internal {
    int requires_reduction(const Array& output, const std::vector<int>& in_bshape);
}

template<typename Class, typename... Args>
struct BaseLazyFunction: public LazyExp<Class> {
    static const int evaluation_dim;
    std::vector<int> bshape_;
    DType dtype_;

    BaseLazyFunction(Args... args);

    static std::vector<int> lazy_output_bshape(const Args&... args);

    static DType lazy_output_dtype(const Args&... args);

    const std::vector<int>& bshape() const;

    const DType& dtype() const;
};

template<typename Class, typename... Args>
struct LazyFunction : public BaseLazyFunction<Class, Args...> {
    using BaseLazyFunction<Class, Args...>::BaseLazyFunction;

    AssignableArray as_assignable() const;
};

template<typename Class, typename... Args>
struct LazyFunctionNonRecusive : public BaseLazyFunction<Class, Args...> {
    using BaseLazyFunction<Class, Args...>::BaseLazyFunction;

    AssignableArray as_assignable() const;
};


#include "dali/array/function/lazy_function-impl.h"

#endif
