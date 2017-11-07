// #ifndef DALI_ARRAY_LAZY_BASE_AXIS_REDUCER_H
// #define DALI_ARRAY_LAZY_BASE_AXIS_REDUCER_H

// #include "dali/array/function/evaluation_dim.h"

// template<typename ExprT>
// static inline auto wrap_3d_around_axis(const ExprT& expr,
//                                        const std::vector<int>& real_expr_shape,
//                                        const int& start_middle,
//                                        const int& end_middle) ->
//         decltype(mshadow::expr::reshape(expr, mshadow::Shape3(1,1,1))) {

//     int before_volume = 1;
//     for (int i = 0; i < start_middle; i++) {
//         before_volume *= real_expr_shape[i];
//     }

//     int middle_volume = 1;
//     for (int i = start_middle; i < end_middle; ++i) {
//         middle_volume *= real_expr_shape[i];
//     }

//     int after_volume = 1;
//     for (int i = end_middle; i < real_expr_shape.size(); i ++) {
//         after_volume *= real_expr_shape[i];
//     }

//     return mshadow::expr::reshape(
//         expr,
//         mshadow::Shape3(before_volume, middle_volume, after_volume)
//     );
// }

// template<class Class, typename ExprT, typename Functor, bool return_indices>
// struct BaseLazyAxisReducer : public LazyFunction<Class, ExprT, int, int, bool> {
//     static const int evaluation_dim;
//     ExprT expr;
//     const int start_reduce;
//     const int end_reduce;
//     const bool keepdims;

//     static std::vector<int> lazy_output_bshape(const ExprT& expr,
//                                                const int& start_reduce_,
//                                                const int& end_reduce_,
//                                                bool keepdims) {
//         auto input_bshape = expr.bshape();
//         std::vector<int> output_bshape;
//         output_bshape.reserve(keepdims ? input_bshape.size() : (input_bshape.size() - 1));
//         for (int i = 0; i < input_bshape.size(); i++) {
//             if (i < start_reduce_ || i >= end_reduce_) {
//                 // if axis is not reduce, keep it
//                 output_bshape.emplace_back(input_bshape[i]);
//             } else if (keepdims) {
//                 // even if axis is reduced, but keepdims is true, we keep
//                 // the axis with dimension 1.
//                 output_bshape.emplace_back(1);
//             }
//         }
//         return output_bshape;
//     }

//     BaseLazyAxisReducer(const ExprT& expr_,
//                         const int& start_reduce_,
//                         const int& end_reduce_,
//                         bool keepdims_) :
//             LazyFunction<Class, ExprT, int, int, bool>(expr_, start_reduce_, end_reduce_, keepdims_),
//             start_reduce(start_reduce_),
//             end_reduce(end_reduce_),
//             keepdims(keepdims_),
//             expr(expr_) {
//         ASSERT2(0 <= start_reduce_ && start_reduce_ < end_reduce_ && end_reduce_ <= expr_.bshape().size(),
//                 utils::MS() << "Reduction axis range ["
//                             << start_reduce_ << ", " << end_reduce_ << ")"
//                             << " must be a nonempty range inside [0, " << expr_.bshape().size() << ").");
//     }

//     template<int devT,typename T, int ndim>
//     auto to_mshadow_expr(memory::Device device,
//                          const std::vector<int>& output_shape,
//                          const lazy::EvaluationSpec<devT, T, ndim>& wrap_array) const ->
//             decltype(
//                 mshadow::expr::reshape(
//                     mshadow::expr::reduce_with_axis<Functor, return_indices>(
//                         wrap_3d_around_axis(
//                             MshadowWrapper<devT, T, decltype(expr)>::wrap(
//                                 expr, device, output_shape,
//                                 wrap_array.template d<lazy::OptimalNdimForInput<ExprT,3>::value>()
//                             ),
//                             output_shape,
//                             start_reduce,
//                             end_reduce
//                         ),
//                         1
//                     ),
//                     mshadow::Shape2(0, 0)
//                 )
//             ) {
//         // Our reduction operates on an expression with the following shape:
//         std::vector<int> new_expr_shape;
//         // (e.g. say we are reducing a matrix of size {Broadcasted, 3, 4} over the
//         // middle axis and saving it to an output location of shape {5, 4}.
//         // In order to prepare our input expression to produce the right
//         // shape, we take the output shape and re-insert the dimensions in
//         // the reduced range.
//         //
//         // For example, reduce of {5,3,4} with add back a 3 in the middle so:
//         // {5, 4} -> {5, 3, 4}
//         //

//         std::vector<int> expr_bshape = expr.bshape();

//         if (keepdims) {
//             // If keepdims was passed as an argument, our output shape already
//             // has the reduced axes present, but with size 1. Instead of inserting
//             // back the necessary reduced axes, we instead replace the dimensions
//             // that were forced to be kept post-reduction
//             // {B, 3, 4} -> reduce -> {5, 1, 4} -> reintroduce middle axis -> {5, 3, 4}
//             // ^^^^^^^^^ expr_bshape  ^^^^^^^^^ output_shape                  ^^^^^^^^^ new_expr_shape
//             new_expr_shape = output_shape;
//             for (int i = start_reduce; i < end_reduce; ++i) {
//                 int dim_size_before_reduction = std::abs(expr_bshape[i]);
//                 new_expr_shape[i] = dim_size_before_reduction;
//             }
//         } else {
//             int reduce_range_length = end_reduce - start_reduce;
//             for (int i = 0; i < expr_bshape.size(); ++i) {
//                 if (i < start_reduce) {
//                     new_expr_shape.emplace_back(output_shape[i]);
//                 } else if (i < end_reduce) {
//                     // if we do not use keepdims, then we can insert at the location
//                     // of the reduction axis, the previous dimension's size:
//                     new_expr_shape.emplace_back(std::abs(expr_bshape[i]));
//                 } else {
//                     new_expr_shape.emplace_back(output_shape[i - reduce_range_length]);
//                 }
//             }
//         }

//         auto wrapped_expr  =
//                 MshadowWrapper<devT, T, decltype(expr)>::wrap(
//                         expr, device, new_expr_shape, wrap_array.template d<lazy::OptimalNdimForInput<ExprT,3>::value>()
//                 );
//         auto result_expr = mshadow::expr::reduce_with_axis<Functor, return_indices>(
//             wrap_3d_around_axis(wrapped_expr, new_expr_shape, start_reduce, end_reduce),
//             1
//         );

//         auto output_flat2D = internal::canonical_reshape<2>(output_shape);
//         return mshadow::expr::reshape(result_expr, mshadow::Shape2(output_flat2D[0], output_flat2D[1]));
//     }
// };

// template<class Class, typename ExprT, typename Functor, bool return_indices>
// const int BaseLazyAxisReducer<Class, ExprT, Functor, return_indices>::evaluation_dim = 2;

// #endif
