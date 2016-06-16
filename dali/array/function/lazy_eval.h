#ifndef DALI_ARRAY_FUNCTION_LAZY_EVAL_H
#define DALI_ARRAY_FUNCTION_LAZY_EVAL_H

#include "dali/array/function/lazy_evaluator.h"
#define DALI_ARRAY_HIDE_LAZY 1
#include "dali/array/array.h"
#undef DALI_ARRAY_HIDE_LAZY

namespace lazy {

	/* In order to make a Lazy Expression be assignable and computable into an array
	or result we must call `lazy::Eval<Outtype>::eval` on the expression.

	The resulting expression is then of two types:

		1) An expression that has a well known shape, for which the resulting computation
		   is known in advance

		2) An expression which may require reductions to be assigned, hence requires
		   modification to lazy expression before application.


	In case (1), we can easily construct the necessary code to run this computation.

	However in case (2), the result of the expression post-reduction is itself a lazy
	expression: because of this mechanism we have:

		lazy_expression -> reduction -> lazy_expression

	hence, an infinite recursion will occur, leading to 'auto-reduce'-code being
	generated for other 'auto-reduce' partial expressions.

	To prevent this recursion from occuring, we can evaluate an expression
	using `lazy::Eval<Outtype>::eval_no_autoreduce`, telling the lazy expression to not generate
	automatic reduction code for the computation, thereby breaking the recursion:

		lazy_expression -> reduction -> unreducible_lazy_expression
	*/

	template<typename OutType>
    struct Eval {
        template<typename Class, typename... Args>
        static inline Assignable<OutType> eval_no_autoreduce(const LazyFunction<Class, Args...>& expr);

        template<typename Class, typename... Args>
        static inline Assignable<OutType> eval(const LazyFunction<Class, Args...>& expr);

        // Here we specialize the lazy evaluation so that the resulting uncomputed expression
        // can not be further reduced (which would lead to infinite recursion => e.g. sum(sum(sum(a))) etc...)
        template<typename Functor, typename ExprT, typename... Args>
        static inline Assignable<OutType> eval(const LazyFunction<LazyAllReducer<Functor, ExprT>, Args...>& expr);
    };
}  // namespace lazy

#include "dali/array/function/lazy_eval-impl.h"

#endif
