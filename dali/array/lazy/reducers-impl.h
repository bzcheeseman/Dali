#include <vector>

#include "dali/array/shape.h"
#include "dali/array/function/lazy_function.h"

#include <mshadow/extension/reduce_with_axis.h>

template<class Functor, typename ExprT>
struct LazyAllReducer : public LazyFunction<LazyAllReducer<Functor,ExprT>, ExprT> {
    static const int evaluation_dim;
    ExprT expr;

    static std::vector<int> lazy_output_bshape(const ExprT&) {
        return {};
    }

    LazyAllReducer(const ExprT& expr_) :
            LazyFunction<LazyAllReducer<Functor,ExprT>, ExprT>(expr_),
            expr(expr_) {
    }

    template<int devT, typename T>
    auto to_mshadow_expr(memory::Device device, const std::vector<int>& output_shape) const ->
            decltype(
                Functor::reduce(
                    MshadowWrapper<devT,T,decltype(expr)>::wrap(expr, device, output_shape)
                )
            ) {

        auto left_expr  =
                MshadowWrapper<devT, T, decltype(expr)>::wrap(
                        expr, device, bshape2shape(expr.bshape())
                );
        auto ret = Functor::reduce(left_expr);
        return ret;
    }
};

template<typename ExprT>
static inline auto wrap_3d_around_axis(const ExprT& expr, const std::vector<int>& real_expr_shape, const int& kept_axis) ->
        decltype(mshadow::expr::reshape(expr, mshadow::Shape3(1,1,1))) {

    int before_size = 1;
    for (int i = 0; i < kept_axis; i++) {
        before_size *= real_expr_shape[i];
    }
    int after_size = 1;
    for (int i = kept_axis + 1; i < real_expr_shape.size(); i ++) {
        after_size *= real_expr_shape[i];
    }

    return mshadow::expr::reshape(expr, mshadow::Shape3(before_size, real_expr_shape[kept_axis], after_size));
}


template<class Functor, typename ExprT>
struct LazyAxisReducer : public LazyFunction<LazyAxisReducer<Functor,ExprT>, ExprT, int> {
    static const int evaluation_dim;
    ExprT expr;
    const int reduce_axis;

    static std::vector<int> lazy_output_bshape(const ExprT& expr, const int& reduce_axis_) {
        auto input_bshape = expr.bshape();
        std::vector<int> output_bshape;
        output_bshape.reserve(input_bshape.size() - 1);
        for (int i = 0; i < input_bshape.size(); i++) {
            if (i != reduce_axis_) {
                output_bshape.emplace_back(input_bshape[i]);
            }
        }
        return output_bshape;
    }

    LazyAxisReducer(const ExprT& expr_, const int& reduce_axis_) :
            LazyFunction<LazyAxisReducer<Functor,ExprT>, ExprT, int>(expr_, reduce_axis_),
            reduce_axis(reduce_axis_),
            expr(expr_) {
        ASSERT2(0 <= reduce_axis && reduce_axis < expr_.bshape().size(),
                utils::MS() << "Reduction axis (" << reduce_axis << ") must be less than input's ndims (" << expr_.bshape().size() << ")");
    }

    template<int devT,typename T>
    auto to_mshadow_expr(memory::Device device, const std::vector<int>& output_shape) const ->
            decltype(
                mshadow::expr::reshape(
                    mshadow::expr::reduce_with_axis<Functor, false>(
                        wrap_3d_around_axis(
                            MshadowWrapper<devT, T, decltype(expr)>::wrap(expr, device, output_shape),
                            output_shape,
                            reduce_axis
                        ),
                        1
                    ),
                    mshadow::Shape2(0, 0)
                )
            ) {
        std::vector<int> new_expr_shape(output_shape);
        new_expr_shape.insert(new_expr_shape.begin() + reduce_axis, std::abs(expr.bshape()[reduce_axis]));
        auto wrapped_expr  =
                MshadowWrapper<devT, T, decltype(expr)>::wrap(
                        expr, device, new_expr_shape
                );

        auto result_expr = mshadow::expr::reduce_with_axis<Functor, false>(
            wrap_3d_around_axis(wrapped_expr, new_expr_shape, reduce_axis),
            1
        );

        auto output_flat2D = internal::canonical_reshape<2>(output_shape);
        return mshadow::expr::reshape(result_expr, mshadow::Shape2(output_flat2D[0], output_flat2D[1]));
    }
};

template<class Functor, typename ExprT>
const int LazyAllReducer<Functor,ExprT>::evaluation_dim = 1;

template<class Functor, typename ExprT>
const int LazyAxisReducer<Functor,ExprT>::evaluation_dim = 2;

namespace myops {
    struct sum_all {
        template<typename T>
        static inline auto reduce(const T& expr) -> decltype(mshadow::expr::sum_all(expr)) {
            return mshadow::expr::sum_all(expr);
        }
    };
}

namespace lazy {
    template<typename ExprT>
    LazyAllReducer<myops::sum_all, ExprT> sum_all(const Exp<ExprT>& expr) {
        return LazyAllReducer<myops::sum_all, ExprT>(expr.self());
    }

    template<typename ExprT>
    LazyAxisReducer<mshadow::red::sum, ExprT> sum_axis(const Exp<ExprT>& expr, const int& reduce_axis) {
        return LazyAxisReducer<mshadow::red::sum, ExprT>(expr.self(), reduce_axis);
    }
}  // namespace lazy
