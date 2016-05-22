#include "dot.h"

#include <vector>

#include "dali/array/function/function.h"
#include "dali/array/shape.h"
#include "dali/array/mshadow_extension/dali_gemm_engine_exp.h"
#include "dali/array/op.h"
#include "dali/array/op/tensordot_as_dot.h"

////////////////////////////////////////////////////////////////////////////////
//                         Matrix multiplication                              //
////////////////////////////////////////////////////////////////////////////////


template<OPERATOR_T operator_t, int devT, typename T>
struct MatrixMultiplyHelper {
    template <
        OPERATOR_T var_operator_t = operator_t,
        typename var_T = T,
        typename std::enable_if<
            !(
                (var_operator_t == OPERATOR_T_MUL) ||
                (var_operator_t == OPERATOR_T_DIV)
            )
        >::type* = nullptr
    >
    static void run(TypedArray<devT, T>& out,
                    const TypedArray<devT, T>& a,
                    const TypedArray<devT, T>& b) {
        typedef decltype(a.contiguous_d2()) mshadow_tensor_t;
        bool             a_transposed, b_transposed;
        mshadow_tensor_t a_tensor,     b_tensor;
        std::tie(a_transposed, a_tensor) = a.blas_friendly_tensor();
        std::tie(b_transposed, b_tensor) = b.blas_friendly_tensor();

        operator_assign_contiguous<operator_t, 2>(
            out,
            dali_gemm(
                a_tensor,
                b_tensor,
                a_transposed,
                b_transposed,
                (T)1.0f
            )
        );
    }

    template <
        OPERATOR_T var_operator_t = operator_t,
        typename var_T = T,
        typename std::enable_if<
            (var_operator_t == OPERATOR_T_MUL) ||
            (var_operator_t == OPERATOR_T_DIV)
        >::type* = nullptr
    >
    static void run(TypedArray<devT, T>& out,
                    const TypedArray<devT, T>& a,
                    const TypedArray<devT, T>& b) {
        ASSERT2(!(var_operator_t == OPERATOR_T_MUL) && !(var_operator_t == OPERATOR_T_MUL),
                "Matrix multiplication's result cannot be inplace-multiplied or inplace-divided.");
        ASSERT2(false, "If asserts above are complete this message should never be displayed");
    }
};


struct ReshapedMatrixMultiplyFunction : public Function<ReshapedMatrixMultiplyFunction, Array, Array, Array, std::vector<int>> {
    static std::vector<int> deduce_output_bshape(
            const Array& a,
            const Array& b,
            const std::vector<int>& output_shape) {
        return output_shape;
    }

    template< int devT, typename T>
    static TypedArray<devT,T> preprocess_out(const TypedArray<devT, T>& out,
                                const std::vector<int>& a_shape,
                                const std::vector<int>& b_shape) {
        // make output look like 2D output
        ASSERT2(a_shape.size() == 2 && b_shape.size() == 2,
                utils::MS() << "MatrixMultiply inputs must be matrices, got a.ndim()="
                            << a_shape.size() << " and b.ndim()="
                            << b_shape.size() << " tensors.");
        ASSERT2(a_shape[1] == b_shape[0],
            utils::MS() << "shapes " << a_shape << " and " << b_shape << " not aligned: "
                        << a_shape[1] << " (dim 1) != " << b_shape[0] << " (dim 0)");
        Array new_out_array = out.array.copyless_reshape(
            {a_shape[0], b_shape[1]}
        );
        return TypedArray<devT,T>(new_out_array, out.device, new_out_array.shape());
    }

    template<OPERATOR_T operator_t, int devT, typename T>
    void typed_eval(
            TypedArray<devT, T> out,
            TypedArray<devT, T> a,
            TypedArray<devT, T> b,
            const std::vector<int>& output_shape) {
        auto new_out = preprocess_out(out, a.array.shape(), b.array.shape());
        MatrixMultiplyHelper<operator_t,devT,T>::run(new_out, a, b);
    }
};

namespace op {
    ////////////////////////////////////////////////////////////////////////////////
    //                     Various flavors of dot products                        //
    ////////////////////////////////////////////////////////////////////////////////

    AssignableArray outer(const Array& a, const Array& b) {
        return ReshapedMatrixMultiplyFunction::run(
            a.reshape({a.number_of_elements(), 1}),
            b.reshape({1, b.number_of_elements()}),
            {a.number_of_elements(), b.number_of_elements()}
        );
    }

    AssignableArray vectordot(
            const Array& a,
            const Array& b) {
        ASSERT2(a.ndim() == 1 && b.ndim() == 1,
            utils::MS() << "VectorDot must be called on a pair of vectors, but got a.ndim()="
                        << a.ndim() << " and b.ndim()=" << b.ndim() << " tensors.");
        ASSERT2(a.bshape()[0] == b.bshape()[0] || (a.bshape()[0] == -1) || (b.bshape()[0] == -1),
            utils::MS() << "shapes " << a.shape() << " and " << b.shape() << " not aligned: "
                        << a.shape()[0] << " (dim 0) != " << b.shape()[0] << " (dim 0)");
        return ReshapedMatrixMultiplyFunction::run(
            a.copyless_reshape({1, a.number_of_elements()}),
            b.copyless_reshape({b.number_of_elements(), 1}),
            {}
        );
    }

    AssignableArray matrixdot(
            const Array& a,
            const Array& b) {
        ASSERT2(a.ndim() == 2 && b.ndim() == 2,
                utils::MS() << "MatrixMultiply inputs must be matrices, got a.ndim()="
                            << a.ndim() << " and b.ndim()="
                            << b.ndim() << " tensors.");
        return ReshapedMatrixMultiplyFunction::run(
                a,
                b,
                {a.shape()[0], b.shape()[1]});
    }

    AssignableArray matrix_vector_dot(
            const Array& a,
            const Array& b) {
        // TODO(jonathan): use correct blas subroutine whenever possible

        ASSERT2((a.ndim() == 1 && b.ndim() == 2) || (a.ndim() == 2 && b.ndim() == 1),
                utils::MS() << "Gemv inputs must be a vector and a matrix, but got a.ndim()="
                            << a.ndim() << " and b.ndim()=" << b.ndim() << " tensors.");
        std::vector<int> outshape(1);
        if (a.ndim() == 1 && b.ndim() == 2) {
            ASSERT2(
                a.bshape()[0] == -1 || b.bshape()[0] == a.bshape()[0] || b.bshape()[0] == -1,
                utils::MS() << "shapes " << a.shape() << " and " << b.shape() << " not aligned: "
                            << a.shape()[0] << " (dim 0) != " << b.shape()[0] << " (dim 0)");
            outshape[0] = b.bshape()[1];
            return ReshapedMatrixMultiplyFunction::run(
                a.copyless_reshape({1, a.number_of_elements()}),
                b,
                outshape
            );
        } else {
            ASSERT2(
                b.bshape()[0] == -1 || a.bshape()[1] == b.bshape()[0] || a.bshape()[0] == -1,
                utils::MS() << "shapes " << a.shape() << " and " << b.shape() << " not aligned: "
                            << a.shape()[1] << " (dim 1) != " << b.shape()[0] << " (dim 0)");
            outshape[0] = a.bshape()[0];
            return ReshapedMatrixMultiplyFunction::run(
                a,
                b.copyless_reshape({b.number_of_elements(), 1}),
                outshape
            );
        }
    }

    template<>
    AssignableArray matrix_multiply_with_reshape(const Array& a,
                                                 const Array& b,
                                                 const std::vector<int>& out_shape) {
        return ReshapedMatrixMultiplyFunction::run(a, b, out_shape);
    }

    AssignableArray tensordot(const Array& a, const Array& b, const int& axis) {
        return tensordot_as_dot(
            a, b, axis, /*batched=*/false
        );
    }

    AssignableArray tensordot(const Array& a, const Array& b, const std::vector<int>& a_reduce_axes, const std::vector<int>& b_reduce_axes) {
        return tensordot_as_dot(
            a, b, a_reduce_axes, b_reduce_axes, /*batched=*/false
        );
    }

////////////////////////////////////////////////////////////////////////////////
//                     The mother of all dot products                         //
////////////////////////////////////////////////////////////////////////////////

    // TODO (szymon): allow for scaling with Binary expression + template redundancy trick!
    AssignableArray dot(const Array& a, const Array& b) {
        auto a_ndim = a.ndim();
        auto b_ndim = b.ndim();

        if (a_ndim == 0 || b_ndim == 0) {
            if (a_ndim == 0) {
                return a.broadcast_scalar_to_ndim(b_ndim) * b;
            } else {
                return a * b.broadcast_scalar_to_ndim(a_ndim);
            }
        } else if (a_ndim > 2 || b_ndim > 2) {
            // a is reduced over the last dimension
            // b is reduced over second to last dimension if it exists,
            // otherwise it is reduced over last.
            return tensordot(a, b, {a_ndim - 1}, {std::max(0, b_ndim - 2)});
        } else if (a_ndim == 1 && b_ndim == 1) {
            return vectordot(a, b);
        } else if (a_ndim == 2 && b_ndim == 2) {
            return matrixdot(a, b);
        } else {
            return matrix_vector_dot(a, b);
        }
    }
}
