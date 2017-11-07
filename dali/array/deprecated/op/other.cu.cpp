// #include "other.h"

// #include <cmath>
// #include <iostream>

// #include "dali/config.h"
// #include "dali/array/array.h"
// #include "dali/array/function/function.h"
// #include "dali/array/memory/device.h"
// #include "dali/array/op2/binary.h"
// #include "dali/array/op2/reducers.h"
// #include "dali/array/op2/unary.h"
// #include "dali/utils/strided_iterator.h"
// #include "dali/utils/make_message.h"

// using memory::Device;

// void operator_eql_check(const OPERATOR_T& operator_t) {
//     ASSERT2(operator_t == OPERATOR_T_EQL, utils::make_message("argsort's result must be "
//         "assigned using operator '=' (note: -=, +=, /=, *= disallowed, got operator = ",
//         operator_t, ")."));
// }

// namespace argsort_helper {

//     template<typename T1, typename T2>
//     void sort_over_args(T1* index_ptr, T2* data_ptr,
//                         const std::vector<int>& shape,
//                         const std::vector<int>& index_strides,
//                         const std::vector<int>& data_strides,
//                         const int& dim = 0) {

//         if (dim + 1 == shape.size()) {
//             auto begin_indices = utils::strided_iterator<T1>(index_ptr, index_strides[dim]);
//             auto end_indices   = begin_indices + shape[dim];
//             auto begin_data    = utils::strided_iterator<T2>(data_ptr, data_strides[dim]);

//             auto assign_index = begin_indices;
//             for (int i = 0; i < shape[dim]; i++) {
//                 *assign_index = i;
//                 assign_index++;
//             }

//             std::sort(begin_indices, end_indices, [begin_data](const T1& lhs, const T1& rhs) {
//                 return *(begin_data + lhs) < *(begin_data + rhs);
//             });
//         } else {
//             // more dims to go
//             for (int idx = 0; idx < shape[dim]; idx++) {
//                 sort_over_args(index_ptr + index_strides[dim] * idx,
//                                data_ptr + data_strides[dim] * idx,
//                                shape,
//                                index_strides,
//                                data_strides,
//                                dim + 1);
//             }
//         }
//     }
// }

// template<OPERATOR_T operator_t, typename T, int devT>
// struct ArgSortFunctionHelper {
//     template <
//         OPERATOR_T var_operator_t = operator_t,
//         typename var_T = T,
//         typename std::enable_if<
//             var_operator_t == OPERATOR_T_EQL
//         >::type* = nullptr
//     >
//     static void run(TypedArray<devT, int>& out,
//                     const TypedArray<devT, T>& in) {

//         argsort_helper::sort_over_args(
//             out.ptr(memory::AM_MUTABLE),
//             in.ptr(),
//             in.array.shape(),
//             out.array.normalized_strides(),
//             in.array.normalized_strides(),
//             0
//         );
//     }

//     template <
//         OPERATOR_T var_operator_t = operator_t,
//         typename var_T = T,
//         typename std::enable_if<!(var_operator_t == OPERATOR_T_EQL)>::type* = nullptr
//     >
//     static void run(TypedArray<devT, int>& out,
//                     const TypedArray<devT, T>& a) {
//         operator_eql_check(operator_t);
//         ASSERT2(false, "If asserts above are complete this message should never be displayed");
//     }
// };

// struct ArgSortFunction : public Function<ArgSortFunction, Array, Array, int> {
//     static std::vector<int> deduce_output_bshape(const Array& arr, const int& axis) {
//         return arr.bshape();
//     };

//     static DType deduce_output_dtype(const Array& arr, const int& axis) {
//         return DTYPE_INT32;
//     }

//     static DType deduce_computation_dtype(const Array& out, const Array& arr, const int& axis) {
//         ASSERT2(out.dtype() == deduce_output_dtype(arr, axis), utils::make_message("ArgSortFunction's "
//             "output must be of type ", deduce_output_dtype(arr, axis), " (got ", out.dtype(), ")."));
//         return arr.dtype();
//     }

//     static memory::Device deduce_output_device(const Array& arr, const int& axis) {
//         return memory::Device::cpu();
//     }

//     static memory::Device deduce_computation_device(const Array& out, const Array& arr, const int& axis) {
//         return memory::Device::cpu();
//     }

//     static void verify(const Array& arr, const int& axis) {
//         ASSERT2(axis >=0 && axis < arr.ndim(), utils::make_message("argsort axis must be non-negative and "
//             "less than the number of dimensions of input (got axis = ", axis, ", with array.ndim() = ", arr.ndim(), ").")
//         );
//     }

//     static std::tuple<Array, Array, int> prepare_output(
//             const OPERATOR_T& operator_t,
//             Array& out,
//             const Array& arr,
//             const int& axis) {
//         operator_eql_check(operator_t);
//         auto output_bshape = deduce_output_bshape(arr, axis);
//         auto output_dtype  = deduce_output_dtype(arr, axis);
//         auto output_device = deduce_output_device(arr, axis);
//         initialize_output_array(out, output_dtype, output_device, &output_bshape);
//         verify(arr, axis);
//         // move axis to last dimension
//         auto arr_rearranged = arr.swapaxes(axis, -1);
//         // add a new tensor that has arange as a value
//         auto out_rearranged = out.swapaxes(axis, -1);
//         // proceed
//         return std::tuple<Array, Array, int>(
//             out_rearranged,
//             arr_rearranged,
//             axis
//         );
//     }

//     template<OPERATOR_T operator_t, typename T>
//     void typed_eval(TypedArray<memory::DEVICE_T_CPU, int> out, TypedArray<memory::DEVICE_T_CPU, T> input, const int& axis) {
//         ArgSortFunctionHelper<operator_t, T, memory::DEVICE_T_CPU>::run(out, input);
//     }

// #ifdef DALI_USE_CUDA
//     template<OPERATOR_T operator_t, typename T>
//     void typed_eval(TypedArray<memory::DEVICE_T_GPU, int> out, TypedArray<memory::DEVICE_T_GPU, T> input, const int& axis) {
//         ASSERT2(false, "I should never be called");
//     }
// #endif
// };

// template<typename T>
// struct FunctionReturnType<ArgSortFunction, T> {
//     typedef int value;
// };

// namespace op {
//     expression::ExpressionGraph any_isnan(const expression::ExpressionGraph& array) {
//         return op::max(op::isnan(array));
//     }

//     expression::ExpressionGraph any_isinf(const expression::ExpressionGraph& array) {
//         return op::max(op::isinf(array));
//     }

//     expression::ExpressionGraph any_isnan(const expression::ExpressionGraph& array, int axis) {
//         if (axis < 0) axis = array.ndim() + axis;
//         return op::max(op::isnan(array), {axis});
//     }

//     expression::ExpressionGraph any_isinf(const expression::ExpressionGraph& array, int axis) {
//         if (axis < 0) axis = array.ndim() + axis;
//         return op::max(op::isinf(array), {axis});
//     }

//     expression::ExpressionGraph all_equals(const expression::ExpressionGraph& left,
//                                       const expression::ExpressionGraph& right) {
//         return op::prod(op::equals(left, right));
//     }

//     expression::ExpressionGraph all_close(const expression::ExpressionGraph& left,
//                                      const expression::ExpressionGraph& right,
//                                      const double& atolerance) {
//         ASSERT2(atolerance >= 0, utils::make_message("atolerance must be a "
//             "strictly positive number (got atolerance=", atolerance, ")."));
//         return op::prod(
//             op::lessthanequal(
//                 op::abs(
//                     op::sub(left,right)
//                 ),
//                 atolerance
//             )
//         );
//     }

//     Assignable<Array> argsort(const Array& array, int axis) {
//         if (axis < 0) axis = array.ndim() + axis;
//         return ArgSortFunction::run(array, axis);
//     }

//     Assignable<Array> argsort(const Array& array) {
//         return ArgSortFunction::run(array.ravel(), 0);
//     }
// } // namespace op

