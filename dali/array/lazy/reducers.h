#ifndef DALI_ARRAY_LAZY_REDUCERS_H
#define DALI_ARRAY_LAZY_REDUCERS_H

#include "dali/array/function/expression.h"

template<class Functor, typename ExprT>
struct LazyAllReducer;

template<class Functor, typename ExprT, bool return_indices>
struct LazyAxisReducer;

template<class Functor, typename ExprT>
struct LazyAllReducerExceptAxis;

namespace mshadow {
    namespace red {
        struct sum;
        struct maximum;
        struct minimum;
        struct product;
    }
};

namespace lazy {
    template<typename ExprT>
    LazyAllReducer<mshadow::red::sum, ExprT> sum(const Exp<ExprT>& expr);

    template<typename ExprT>
    LazyAllReducerExceptAxis<mshadow::red::sum, ExprT> sumall_except(const Exp<ExprT>& expr, const int& axis, bool keepdims=false);

    template<typename ExprT>
    LazyAllReducer<mshadow::red::minimum, ExprT> min(const Exp<ExprT>& expr);

    template<typename ExprT>
    LazyAllReducer<mshadow::red::maximum, ExprT> max(const Exp<ExprT>& expr);

    template<typename ExprT>
    LazyAllReducer<mshadow::red::product, ExprT> product(const Exp<ExprT>& expr);






    #define DALI_DECLARE_LAZY_ARRAY_AXIS_REDUCER(OPNAME, REDUCERNAME, RETURN_INDICES)\
        template<typename ExprT>\
        LazyAxisReducer<REDUCERNAME, ExprT, RETURN_INDICES> OPNAME(const Exp<ExprT>& expr,\
                                                                   const int& axis, \
                                                                   bool keepdims=false); \
        template<typename ExprT>\
        LazyAxisReducer<REDUCERNAME, ExprT, RETURN_INDICES> OPNAME(const Exp<ExprT>& expr, \
                                                                   const int& redude_start, \
                                                                   const int& redude_end, \
                                                                   bool keepdims=false);

    DALI_DECLARE_LAZY_ARRAY_AXIS_REDUCER(sum,     mshadow::red::sum,     false);
    DALI_DECLARE_LAZY_ARRAY_AXIS_REDUCER(product, mshadow::red::product, false);
    DALI_DECLARE_LAZY_ARRAY_AXIS_REDUCER(min,     mshadow::red::minimum, false);
    DALI_DECLARE_LAZY_ARRAY_AXIS_REDUCER(max,     mshadow::red::maximum, false);
    DALI_DECLARE_LAZY_ARRAY_AXIS_REDUCER(argmax,  mshadow::red::maximum, true);
    DALI_DECLARE_LAZY_ARRAY_AXIS_REDUCER(argmin,  mshadow::red::minimum, true);



}  // namespace lazy

#include "dali/array/lazy/reducers-impl.h"

#endif
