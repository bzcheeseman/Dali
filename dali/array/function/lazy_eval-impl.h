#include "dali/array/lazy/reducers.h"

namespace internal {
    struct ReductionInstruction {
        bool required = false;
        int axis = -1;
        bool all_reduce = false;
    };

    // deduce how many reductions need to be made, and if an all-reduce could achieve it
    ReductionInstruction requires_reduction(const Array& output, const std::vector<int>& in_bshape);
}

namespace lazy {
    template<typename Class, typename... Args>
    inline AssignableArray eval_no_autoreduce(const LazyFunction<Class, Args...>& expr) {
        return LazyEvaluator<Class>::run(expr.self());
    }

    // Here we specialize the lazy evaluation so that the resulting uncomputed expression
    // can not be further reduced (which would lead to infinite recursion => e.g. sum(sum(sum(a))) etc...)
    template<typename Functor, typename ExprT, typename... Args>
    inline AssignableArray eval(const LazyFunction<LazyAllReducer<Functor, ExprT>, Args...>& expr) {
        return eval_no_autoreduce(expr);
    }

    template<typename Class, typename... Args>
    inline AssignableArray eval(const LazyFunction<Class, Args...>& expr) {
        auto this_self   = expr.self();
        auto this_bshape = expr.bshape();

        return AssignableArray([this_self, this_bshape](Array& out, const OPERATOR_T& operator_t) {
            internal::ReductionInstruction reduction_dimension;
            if (operator_t == OPERATOR_T_LSE) {
                reduction_dimension = internal::requires_reduction(out, this_bshape);
            }
            if (reduction_dimension.axis != -1) {
                auto reduced_expr = lazy::sum(
                       this_self,
                       /*axis=*/reduction_dimension.axis,
                       /*keepdims=*/true);
                auto computation_with_reduce = lazy::eval_no_autoreduce(reduced_expr);
                computation_with_reduce.assign_to(out, operator_t);
            } else if (reduction_dimension.all_reduce) {
                auto reduced_expr = lazy::sum(this_self);
                auto computation_with_reduce = lazy::eval_no_autoreduce(reduced_expr);
                auto out_as_scalar = out.copyless_reshape({});
                computation_with_reduce.assign_to(out_as_scalar, operator_t);
            } else {
                LazyEvaluator<Class>::run(this_self).assign_to(out, operator_t);
            }
        });
    }

}  // namespace lazy
