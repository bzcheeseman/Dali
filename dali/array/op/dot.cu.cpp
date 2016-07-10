#include "dot.h"

#include <vector>

#include "dali/array/array.h"
#include "dali/array/function/function.h"
#include "dali/array/shape.h"
#include "dali/array/mshadow_extension/dali_gemm_engine_exp.h"
#include "dali/array/op.h"
#include "dali/array/op/tensordot_as_dot.h"
#include "dali/array/shape.h"

////////////////////////////////////////////////////////////////////////////////
//                         Matrix multiplication                              //
////////////////////////////////////////////////////////////////////////////////

template<OPERATOR_T operator_t, typename T, int devT>
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


struct MatrixMultiplyFunction : public Function<MatrixMultiplyFunction,
                                                Array,
                                                Array,
                                                Array> {
    static std::vector<int> deduce_output_bshape(
            const Array& a,
            const Array& b) {
        return {a.bshape()[0], b.bshape()[1]};
    }


    static std::tuple<Array, Array, Array> prepare_output(
            const OPERATOR_T& operator_t,
            Array& out,
            const Array& left,
            const Array& right) {
        auto output_bshape = deduce_output_bshape(left, right);
        auto output_dtype  = deduce_output_dtype(left, right);
        auto output_device = deduce_output_device(left, right);

        initialize_output_array(out, output_dtype, output_device, &output_bshape);

        verify(left, right);

        int left_dim  = std::abs(out.shape()[0]),
            inner_dim = std::max(left.shape()[1], right.shape()[0]),
            right_dim = std::abs(out.shape()[1]);

        Array left_reshaped  = left.reshape_broadcasted({left_dim, inner_dim});
        if (!left_reshaped.is_transpose()) {
            left_reshaped = left_reshaped.ascontiguousarray();
        }

        Array right_reshaped  = right.reshape_broadcasted({inner_dim, right_dim});
        if (!right_reshaped.is_transpose()) {
            right_reshaped = right_reshaped.ascontiguousarray();
        }

        Array out_reshaped  = out.reshape_broadcasted({left_dim, right_dim});
        if (!out_reshaped.is_transpose()) {
            // TODO(szymon): change this to reshape (not copyless) and make sure
            // that if the copy has occured we write back the result to out after
            // computation is done.
            // TODO(szymon): make sure that the test ArrayDotTests.broadcast_to_out
            // does make a copy before assigning to out, so that the computation is
            // 3x less
            ASSERT2(out_reshaped.contiguous_memory(),
                    "assigning the result of matrix multiplication to array with noncontiguous memory is currently unsupported.");
        }

        ASSERT2(out_reshaped.memory() != right_reshaped.memory() &&
                out_reshaped.memory() != left_reshaped.memory(),
            "inplace matrix multiply is not supported at the time.");

        if (out_reshaped.is_transpose()) {
            // some blas implementations do not allow tranposed output.
            // we get around it by using a basic linear algebra identity:
            //
            //  C = (A dot B)
            //
            // is equivalent to
            //
            //  C^T = (A dot B)^T = B^T dot A^T

            return std::tuple<Array,Array,Array>(out_reshaped.transpose(),
                                                 right_reshaped.transpose(),
                                                 left_reshaped.transpose());
        } else {
            return std::tuple<Array,Array,Array>(out_reshaped, left_reshaped, right_reshaped);
        }
    }

    static bool dimensions_compatible(int dim1, int dim2) {
        return dim1 == dim2 || dim1 == -1 || dim2 == -1;
    }

    static void verify(const Array& a, const Array& b) {
        ASSERT2(a.is_matrix() && b.is_matrix(),
                utils::MS() << "Matrix multiply inputs must be matrices, got a.ndim()="
                            << a.ndim() << " and b.ndim()="
                            << b.ndim() << " tensors.");

        ASSERT2(dimensions_compatible(a.bshape()[1], b.bshape()[0]),
            utils::MS() << "Matrix multiply dissagreement on inner dimension for shapes "
                        << a.shape() << " and " << b.shape());
    }

    template<OPERATOR_T operator_t, typename T, int devT>
    void typed_eval(
            TypedArray<devT, T> out,
            TypedArray<devT, T> left,
            TypedArray<devT, T> right) {

        MatrixMultiplyHelper<operator_t,T,devT>::run(out, left, right);
    }
};


struct ReshapedMatrixMultiplyFunction : public Function<ReshapedMatrixMultiplyFunction,
                                                        Array,
                                                        Array,
                                                        Array,
                                                        std::vector<int>,
                                                        std::vector<int>> {
    static std::vector<int> deduce_output_bshape(
            const Array& a,
            const Array& b,
            const std::vector<int>& output_shape,
            const std::vector<int>& output_shape_2d) {
        return output_shape;
    }

    template<OPERATOR_T operator_t, typename T, int devT>
    void typed_eval(
            TypedArray<devT, T> out,
            TypedArray<devT, T> a,
            TypedArray<devT, T> b,
            const std::vector<int>& output_shape,
            const std::vector<int>& output_shape_2d) {
        // Comprehensive error checking
        ASSERT2(a.array.is_matrix() && b.array.is_matrix(),
                utils::MS() << "MatrixMultiply inputs must be matrices, got a.ndim()="
                            << a.array.ndim() << " and b.ndim()="
                            << b.array.ndim() << " tensors.");
        ASSERT2(output_shape_2d.size() == 2,
                utils::MS() << "MatrixMultiply: 2D output shape must be 2D (got "
                            << output_shape_2d << " )");

        ASSERT2(output_shape_2d[0] == a.array.shape()[0] && output_shape_2d[1] == b.array.shape()[1],
                utils::MS() << "MatrixMultiply: matrix of shape " << a.array.shape()
                            << "multiplied by matrix of shape " << b.array.shape()
                            << "does not result in suggested output_shape_2d ("
                            << output_shape_2d << ").");
        ASSERT2(a.array.shape()[1] == b.array.shape()[0],
            utils::MS() << "MatrixMultiply: dissagreement on inner dimension for shapes "
                        << a.array.shape() << " and " << b.array.shape());

        ASSERT2(hypercube_volume(output_shape_2d) == hypercube_volume(output_shape),
                utils::MS() << "MatrixMultiply: output_shape_2d (" << output_shape_2d
                            << ") must have the same number of elements as output_shape ("
                            << output_shape << ")");

        // output reshaping
        Array new_out_array = out.array.copyless_reshape(output_shape_2d);

        auto new_out = TypedArray<devT,T>(new_out_array, out.device, new_out_array.shape());

        MatrixMultiplyHelper<operator_t,T,devT>::run(new_out, a, b);
    }
};

static Assignable<Array> reshaped_matmul_fix_broadcasts(const Array& a,
                                                      const Array& b,
                                                      const std::vector<int>& output_shape,
                                                      const std::vector<int>& output_shape_2d) {
    // TODO(szymon): this should be generalized to all dali functions.
    // Similar logic already happens in lazy, so maybe it can be leveraged.
    ASSERT2(a.is_matrix() && b.is_matrix(),
            utils::MS() << "MatrixMultiply inputs must be matrices, got a.ndim()="
                        << a.shape().size() << " and b.ndim()="
                        << b.shape().size() << " tensors.");
    ASSERT2(output_shape_2d.size() == 2,
            utils::MS() << "MatrixMultiply: 2D output shape must be 2D (got "
                        << output_shape_2d << " )");

    auto a_bshape = a.bshape();
    auto b_bshape = b.bshape();
    int left   = output_shape_2d[0],
        middle = std::max(a.shape()[1], b.shape()[0]),
        right  = output_shape_2d[1];
    // if the broadcasting fails let ReshapedMatrixMultiplyFunction
    // throw an error.
    Array new_a = a, new_b = b;
    try {
        new_a = a.reshape_broadcasted({left, middle}).ascontiguousarray();
    } catch(std::runtime_error) {}

    try {
        new_b = b.reshape_broadcasted({middle, right}).ascontiguousarray();
    } catch(std::runtime_error) {}

    return ReshapedMatrixMultiplyFunction::run(
        new_a,
        new_b,
        output_shape,
        output_shape_2d
    );
}

static Assignable<Array> reshaped_matmul_fix_broadcasts(const Array& a,
                                                      const Array& b,
                                                      const std::vector<int>& output_shape) {
    return reshaped_matmul_fix_broadcasts(a,b,output_shape, output_shape);
}

namespace op {
    ////////////////////////////////////////////////////////////////////////////////
    //                     Various flavors of dot products                        //
    ////////////////////////////////////////////////////////////////////////////////

    Assignable<Array> outer(const Array& a, const Array& b) {
        return reshaped_matmul_fix_broadcasts(
            a.reshape({a.number_of_elements(), 1}),
            b.reshape({1, b.number_of_elements()}),
            {a.number_of_elements(), b.number_of_elements()}
        );
    }

    Assignable<Array> inner(
            const Array& a,
            const Array& b) {
        ASSERT2(a.ndim() == 1 && b.ndim() == 1,
            utils::MS() << "inner must be called on a pair of vectors, but got a.ndim()="
                        << a.ndim() << " and b.ndim()=" << b.ndim() << " tensors.");
        ASSERT2(a.bshape()[0] == b.bshape()[0] || (a.bshape()[0] == -1) || (b.bshape()[0] == -1),
            utils::MS() << "shapes " << a.shape() << " and " << b.shape() << " not aligned: "
                        << a.shape()[0] << " (dim 0) != " << b.shape()[0] << " (dim 0)");
        return reshaped_matmul_fix_broadcasts(
            a.copyless_reshape({1, a.number_of_elements()}),
            b.copyless_reshape({b.number_of_elements(), 1}),
            {},
            {1,1}
        );
    }

    Assignable<Array> matrixdot(
            const Array& a,
            const Array& b) {
        return MatrixMultiplyFunction::run(a, b);
    }

    Assignable<Array> matrix_vector_dot(
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
            return reshaped_matmul_fix_broadcasts(
                a.copyless_reshape({1, a.number_of_elements()}),
                b,
                outshape,
                {1, outshape[0]}
            );
        } else {
            ASSERT2(
                b.bshape()[0] == -1 || a.bshape()[1] == b.bshape()[0] || a.bshape()[0] == -1,
                utils::MS() << "shapes " << a.shape() << " and " << b.shape() << " not aligned: "
                            << a.shape()[1] << " (dim 1) != " << b.shape()[0] << " (dim 0)");
            outshape[0] = a.bshape()[0];
            return reshaped_matmul_fix_broadcasts(
                a,
                b.copyless_reshape({b.number_of_elements(), 1}),
                outshape,
                {outshape[0], 1}
            );
        }
    }

    template<>
    Assignable<Array> matrix_multiply_with_reshape(const Array& a,
                                                 const Array& b,
                                                 const std::vector<int>& out_shape,
                                                 const std::vector<int>& out_shape_2d) {
        return reshaped_matmul_fix_broadcasts(a, b, out_shape, out_shape_2d);
    }

    Assignable<Array> tensordot(const Array& a, const Array& b, const int& axis) {
        return tensordot_as_dot(
            a, b, axis, /*batched=*/false
        );
    }

    Assignable<Array> tensordot(const Array& a, const Array& b, const std::vector<int>& a_reduce_axes, const std::vector<int>& b_reduce_axes) {
        return tensordot_as_dot(
            a, b, a_reduce_axes, b_reduce_axes, /*batched=*/false
        );
    }

////////////////////////////////////////////////////////////////////////////////
//                     The mother of all dot products                         //
////////////////////////////////////////////////////////////////////////////////

    // TODO (szymon): allow for scaling with Binary expression + template redundancy trick!
    Assignable<Array> dot(const Array& a, const Array& b) {
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
            return inner(a, b);
        } else if (a_ndim == 2 && b_ndim == 2) {
            return matrixdot(a, b);
        } else {
            return matrix_vector_dot(a, b);
        }
    }
}
