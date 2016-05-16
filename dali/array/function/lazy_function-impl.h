#include "dali/array/function/lazy_evaluator.h"
#include "dali/array/function/args/reduce_over_args.h"
#include "dali/array/function/args/property_reducer.h"
#include "dali/array/lazy/base_lazy_axis_reducer.h"

namespace internal {
    template<typename ExprT>
    struct NonRecursiveLazySumAxis : public BaseLazyAxisReducer<LazyFunctionNonRecusive,NonRecursiveLazySumAxis<ExprT>, ExprT, mshadow::red::sum, false> {
        using BaseLazyAxisReducer<LazyFunctionNonRecusive,NonRecursiveLazySumAxis<ExprT>, ExprT, mshadow::red::sum, false>::BaseLazyAxisReducer;
    };
}  // namespace internal



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


template<typename Class, typename... Args>
AssignableArray LazyFunction<Class,Args...>::as_assignable() const {
    auto this_self   = this->self();
    auto this_bshape = this->bshape();

    return AssignableArray([this_self, this_bshape](Array& out, const OPERATOR_T& operator_t) {
        int reduction_dimension = -1;
        if (operator_t == OPERATOR_T_LSE) {
            reduction_dimension = internal::requires_reduction(out, this_bshape);
        }
        if (reduction_dimension != -1) {
            auto reduced_expr = internal::NonRecursiveLazySumAxis<decltype(this_self)>(
                    this_self,
                    /*axis=*/reduction_dimension,
                    /*keepdims=*/true);
            auto computation_with_reduce = reduced_expr.as_assignable();
            computation_with_reduce.assign_to(out, operator_t);
        } else {
            LazyEvaluator<Class>::run(this_self).assign_to(out, operator_t);
        }
    });
}


template<typename Class, typename... Args>
AssignableArray LazyFunctionNonRecusive<Class,Args...>::as_assignable() const {
    return LazyEvaluator<Class>::run(this->self());
}
