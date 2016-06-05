#ifndef DALI_ARRAY_LAZY_BASE_AXIS_REDUCER_H
#define DALI_ARRAY_LAZY_BASE_AXIS_REDUCER_H

#include "dali/array/function/evaluation_dim.h"

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

template<class Class, typename ExprT, typename Functor, bool return_indices>
struct BaseLazyAxisReducer : public LazyFunction<Class, ExprT, int, bool> {
    static const int evaluation_dim;
    ExprT expr;
    const int reduce_axis;
    const bool keepdims;

    static std::vector<int> lazy_output_bshape(const ExprT& expr, const int& reduce_axis_, bool keepdims) {
        auto input_bshape = expr.bshape();
        std::vector<int> output_bshape;
        output_bshape.reserve(keepdims ? input_bshape.size() : (input_bshape.size() - 1));
        for (int i = 0; i < input_bshape.size(); i++) {
            if (i != reduce_axis_) {
                // if axis is not reduce, keep it
                output_bshape.emplace_back(input_bshape[i]);
            } else if (keepdims) {
                // even if axis is reduced, but keepdims is true, we keep
                // the axis with dimension 1.
                output_bshape.emplace_back(1);
            }
        }
        return output_bshape;
    }

    BaseLazyAxisReducer(const ExprT& expr_, const int& reduce_axis_, bool keepdims_) :
            LazyFunction<Class, ExprT, int, bool>(expr_, reduce_axis_, keepdims_),
            reduce_axis(reduce_axis_),
            keepdims(keepdims_),
            expr(expr_) {
        ASSERT2(0 <= reduce_axis && reduce_axis < expr_.bshape().size(),
                utils::MS() << "Reduction axis (" << reduce_axis << ") must be less than input's ndims (" << expr_.bshape().size() << ")");
    }

    template<int devT,typename T, int ndim>
    auto to_mshadow_expr(memory::Device device,
                         const std::vector<int>& output_shape,
                         const lazy::EvaluationSpec<devT, T, ndim>& wrap_array) const ->
            decltype(
                mshadow::expr::reshape(
                    mshadow::expr::reduce_with_axis<Functor, return_indices>(
                        wrap_3d_around_axis(
                            MshadowWrapper<devT, T, decltype(expr)>::wrap(
                                expr, device, output_shape,
                                wrap_array.template d<lazy::OptimalNdimForInput<ExprT,3>::value>()
                            ),
                            output_shape,
                            reduce_axis
                        ),
                        1
                    ),
                    mshadow::Shape2(0, 0)
                )
            ) {
        // Our reduction operates on an expression with the following shape:
        // (e.g. say we are reducing a matrix of size {3,4} over the last axis
        // thus we expect our output to have shape {3,})
        std::vector<int> new_expr_shape(output_shape);
        // in order to prepare our input expression to produce the right
        // reduction, we re-insert the dimension that was reduced
        // (e.g. reduce of {3,4} with add back a 4 at the end so: {3,} -> {3,4})
        int input_expr_reduced_axis_size = std::abs(expr.bshape()[reduce_axis]);
        // if keepdims was passed as an argument, our output shape already
        // has the reduced axis present, but with size 1. Instead of inserting
        // back the necessary reduce axis, we instead replace the dimension
        // that was forced to be kept post-reduction
        // (e.g. {3,4} -> reduce -> {3,} -> keepdims -> {3,1} => reintroduce last axis => {3,4})
        if (keepdims) {
            new_expr_shape[reduce_axis] = input_expr_reduced_axis_size;
        } else {
            // if we do not use keepdims, then we can simply insert at the location
            // of the reduction axis, the previous dimension's size:
            new_expr_shape.insert(new_expr_shape.begin() + reduce_axis, input_expr_reduced_axis_size);
        }

        auto wrapped_expr  =
                MshadowWrapper<devT, T, decltype(expr)>::wrap(
                        expr, device, new_expr_shape, wrap_array.template d<lazy::OptimalNdimForInput<ExprT,3>::value>()
                );
        auto result_expr = mshadow::expr::reduce_with_axis<Functor, return_indices>(
            wrap_3d_around_axis(wrapped_expr, new_expr_shape, reduce_axis),
            1
        );

        auto output_flat2D = internal::canonical_reshape<2>(output_shape);
        return mshadow::expr::reshape(result_expr, mshadow::Shape2(output_flat2D[0], output_flat2D[1]));
    }
};

template<class Class, typename ExprT, typename Functor, bool return_indices>
const int BaseLazyAxisReducer<Class, ExprT, Functor, return_indices>::evaluation_dim = 2;

#endif
