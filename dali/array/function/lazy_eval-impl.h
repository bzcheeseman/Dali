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

    // Eval with operator (only instantiate a single operator)

    template<OPERATOR_T intented_operator_t>
    struct EvalWithOperator {
        template<typename Class, typename... Args>
        static inline AssignableArray eval_no_autoreduce(const LazyFunction<Class, Args...>& expr) {
            return LazyEvaluator<Class>::template run_with_operator<intented_operator_t>(expr.self());
        }

        template<typename Class, typename... Args>
        static inline AssignableArray eval(const LazyFunction<Class, Args...>& expr) {
            auto this_self   = expr.self();
            auto this_bshape = expr.bshape();

            return AssignableArray([this_self, this_bshape](Array& out, const OPERATOR_T& operator_t) {
                ASSERT2(operator_t == intented_operator_t,
                    utils::MS() << "AssignableArray constructed for operator "
                                << operator_to_name(intented_operator_t)
                                << " but got " << operator_to_name(operator_t)
                                << " instead");
                LazyEvaluator<Class>::template run_with_operator<intented_operator_t>(
                    this_self
                ).assign_to(
                    out,
                    intented_operator_t
                );
            });
        }

        template<typename Functor, typename ExprT, typename... Args>
        static inline AssignableArray eval(const LazyFunction<LazyAllReducer<Functor, ExprT>, Args...>& expr) {
            return EvalWithOperator<intented_operator_t>::eval_no_autoreduce(expr);
        }

        template<typename Class, typename... Args>
        static inline void assign(const ArraySubtensor& destination, const LazyFunction<Class, Args...>& expr) {
            return LazyEvaluator<Class>::template subtensor_assign_with_operator<intented_operator_t>(destination, expr.self());
        }
    };

    template<>
    struct EvalWithOperator<OPERATOR_T_LSE> {
        template<typename Class, typename... Args>
        static inline AssignableArray eval(const LazyFunction<Class, Args...>& expr) {
            auto this_self   = expr.self();
            auto this_bshape = expr.bshape();

            return AssignableArray([this_self, this_bshape](Array& out, const OPERATOR_T& operator_t) {
                ASSERT2(operator_t == OPERATOR_T_LSE,
                    utils::MS() << "AssignableArray constructed for operator "
                                << operator_to_name(OPERATOR_T_LSE)
                                << " but got " << operator_to_name(operator_t)
                                << " instead");
                auto reduction_dimension = internal::requires_reduction(
                    out,
                    this_bshape
                );
                if (reduction_dimension.axis != -1) {
                    auto reduced_expr = lazy::sum(
                           this_self,
                           /*axis=*/reduction_dimension.axis,
                           /*keepdims=*/true);
                    auto computation_with_reduce = EvalWithOperator<OPERATOR_T_LSE>::eval_no_autoreduce(
                        reduced_expr
                    );
                    computation_with_reduce.assign_to(
                        out,
                        OPERATOR_T_LSE
                    );
                } else if (reduction_dimension.all_reduce) {
                    auto reduced_expr = lazy::sum(this_self);
                    auto computation_with_reduce = EvalWithOperator<OPERATOR_T_LSE>::eval_no_autoreduce(
                        reduced_expr
                    );
                    auto out_as_scalar = out.copyless_reshape({});
                    computation_with_reduce.assign_to(
                        out_as_scalar,
                        OPERATOR_T_LSE
                    );
                } else {
                    LazyEvaluator<Class>::template run_with_operator<OPERATOR_T_LSE>(
                        this_self
                    ).assign_to(
                        out,
                        OPERATOR_T_LSE
                    );
                }
            });
        }

        template<typename Class, typename... Args>
        static inline AssignableArray eval_no_autoreduce(const LazyFunction<Class, Args...>& expr) {
            return LazyEvaluator<Class>::template run_with_operator<OPERATOR_T_LSE>(expr.self());
        }

        template<typename Functor, typename ExprT, typename... Args>
        static inline AssignableArray eval(const LazyFunction<LazyAllReducer<Functor, ExprT>, Args...>& expr) {
            return EvalWithOperator<OPERATOR_T_LSE>::eval_no_autoreduce(expr);
        }
    };

}  // namespace lazy
