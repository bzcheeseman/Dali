#include "dali/array/function/lazy_function.h"

template<typename ExprT, typename NewT>
struct LazyCast : public LazyFunction<LazyCast<ExprT, NewT>, ExprT> {
    ExprT expr;
    static const int evaluation_dim;

    LazyCast(ExprT expr_) : LazyFunction<LazyCast<ExprT, NewT>, ExprT>(expr_),
                             expr(expr_) {
    }

    template<int devT, typename T, int ndim>
    auto to_mshadow_expr(memory::Device device, const std::vector<int>& output_shape, const lazy::EvaluationSpec<devT, T, ndim>& wrap_array) const ->
            decltype(
                mshadow::expr::tcast<NewT>(
                     MshadowWrapper<devT,T,ExprT>::wrap(expr, device, output_shape, wrap_array)
                )
            ) {
        auto left_expr = MshadowWrapper<devT,T,ExprT>::wrap(expr, device, output_shape, wrap_array);
        return mshadow::expr::tcast<NewT>(left_expr);
    }
};

template<typename ExprT, typename NewT>
const int LazyCast<ExprT, NewT>::evaluation_dim = lazy::LazyEvaluationDim<ExprT>::value;

namespace lazy {
    template<typename NewType, typename ExprT>
    LazyCast<ExprT, NewType> astype(const Exp<ExprT>& expr) {
        return LazyCast<ExprT,NewType>(expr.self());
    }
}
