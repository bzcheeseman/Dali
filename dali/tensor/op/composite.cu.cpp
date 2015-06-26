#include "dali/tensor/op/composite.h"

#include "dali/tensor/__MatMacros__.h"
#include "dali/math/TensorOps.h"
#include "dali/math/LazyTensor.h"
#include "dali/tensor/Weights.h"

using std::vector;


namespace matops {
    template<typename R>
    Mat<R> Composite<R>::quadratic_form(
            Mat<R> left,
            Mat<R> middle,
            Mat<R> right) {
        ASSERT2(middle.dims(1) == right.dims(0), "Quadratic form right matrix has wrong dimensions.");
        ASSERT2(left.dims(0) == middle.dims(0) , "Quadratic form left matrix has wrong dimensions.");

        Mat<R> out (left.dims(1), right.dims(1), weights<R>::empty());

        if (graph::backprop_enabled) {
            TensorInternal<R,2> left_side_mul(mshadow::Shape2(left.dims(1), middle.dims(1)));
            // left_side_mul = left.T * middle;
            left_side_mul = dot(MAT(left).wrapper().T(), MAT(middle).wrapper());
            // out = left_side_mul * right;
            MAT(out) = dot(left_side_mul.wrapper(), MAT(right).wrapper());
            graph::emplace_back([left_side_mul, left, middle, right, out]() mutable {
                // right.dw() += left_side_mul.T() * out.dw();
                SAFE_GRAD(right) += dot(left_side_mul.wrapper().T(), GRAD(out).wrapper());
                TensorInternal<R,2> LeftT_dot_middle_grad(mshadow::Shape2(left.dims(1), right.dims(0)));
                // leftT_dot_middle_grad = out.dw() * right.T()
                LeftT_dot_middle_grad = dot(GRAD(out).wrapper(), MAT(right).wrapper().T());
                // left.dw() += leftT_dot_middle_grad * middle.T()
                SAFE_GRAD(left) += dot(MAT(middle).wrapper(), LeftT_dot_middle_grad.wrapper().T());
                // middle.dw() += left * LeftT_dot_middle_grad
                SAFE_GRAD(middle) += dot(MAT(left).wrapper(), LeftT_dot_middle_grad.wrapper());
            });
        } else {
            MAT(out) = MAT(left).wrapper().T() * MAT(middle).wrapper() * MAT(right).wrapper();
        }
        return out;
    }

    template<typename R>
    Mat<R> Composite<R>::mul_with_bias(
            Mat<R> matrix1,
            Mat<R> matrix2,
            Mat<R> bias) {
        ASSERT2(matrix1.dims(1) == matrix2.dims(0), "matmul dimensions misaligned.");
        ASSERT2(matrix1.dims(0) != bias.dims(1) || bias.dims(1) != 1,
            "Matrices cannot be multiplied with broadcast, they do not have the same dimensions.");
        Mat<R> out(matrix1.dims(0), matrix2.dims(1), weights<R>::empty());
        MAT(out) = dot(MAT(matrix1).wrapper(), MAT(matrix2).wrapper());
        MAT(out) += MAT(bias).wrapper()[0].template broadcast<0>(MAT(out).shape());

        if (graph::backprop_enabled)
            graph::emplace_back([matrix1, matrix2, bias, out]() mutable {
                SAFE_GRAD(matrix1) += dot(GRAD(out).wrapper(),        MAT(matrix2).wrapper().T());
                SAFE_GRAD(matrix2) += dot(MAT(matrix1).wrapper().T(), GRAD(out).wrapper());
                SAFE_GRAD(bias).wrapper()[0] += (
                    sum_cols(GRAD(out).wrapper())
                );
            });
        return out;
    }

    template<typename R>
    Mat<R> Composite<R>::mul_add_broadcast_mul_with_bias(
            Mat<R> matrix1,
            Mat<R> input_to_1,
            Mat<R> matrix2,
            Mat<R> input_to_2,
            Mat<R> bias) {
        ASSERT2(matrix1.dims(1) == input_to_1.dims(0), "matmul 1 dimensions misaligned.");
        ASSERT2(matrix2.dims(1) == input_to_2.dims(0), "matmul 2 dimensions misaligned.");
        ASSERT2(matrix2.dims(0) == bias.dims(0) && matrix1.dims(0) == bias.dims(1) && input_to_1.dims(1) == 1 && bias.dims(0) == 1,
            "Matrices cannot be operated on together, they do not have the same output dimensions.");

        Mat<R> out (matrix1.dims(0), input_to_2.dims(1), weights<R>::empty());
        MAT(out) = MAT(bias).wrapper()[0].template broadcast<0>(MAT(out).shape());
        MAT(out) += dot(MAT(matrix2).wrapper(), MAT(input_to_2).wrapper());

        MAT(out) += dot(MAT(matrix1).wrapper(), MAT(input_to_1).wrapper());

        // both input to 1 and bias are columns,
        // so we add both of those before adding the true matrix
        // product in broadcasted form
        {
            TensorInternal<R, 2> temp(mshadow::Shape2(1, input_to_2.dims(1)));
            temp = dot(MAT(matrix1).wrapper(), MAT(input_to_1).wrapper());
            MAT(out) += temp.wrapper()[0].template broadcast<0>(MAT(out).shape());
        }

        if (graph::backprop_enabled)
            graph::emplace_back([matrix1, input_to_1, matrix2, input_to_2, bias, out] () mutable {
                // first multiply:
                // broadcasting input means taking outer product here:

                if (!matrix1.constant) {
                    TensorInternal<R, 2> temp(mshadow::Shape2(1, input_to_2.dims(1)));
                    temp.wrapper()[0] = sum_cols(GRAD(out).wrapper());
                    GRAD(matrix1) += dot(temp.wrapper(), MAT(input_to_1).wrapper().T());
                }

                // broadcasting output means sum after the reverse product here:
                if (!input_to_1.constant) {
                    TensorInternal<R, 2> temp(mshadow::Shape2(matrix1.dims(0), input_to_2.dims(1)));
                    temp = dot(MAT(matrix1).wrapper().T(), GRAD(out).wrapper());
                    GRAD(input_to_1).wrapper()[0] += sum_cols(temp.wrapper());
                }


                // broadcasting input means taking outer product here:
                SAFE_GRAD(matrix2) += dot(GRAD(out).wrapper(), MAT(input_to_2).wrapper().T());
                // broadcasting output means sum after the reverse product here:
                SAFE_GRAD(input_to_2) += dot(MAT(matrix2).wrapper().T(), GRAD(out).wrapper());
                // bias vector:
                SAFE_GRAD(bias).wrapper()[0] += sum_cols(GRAD(out).wrapper());
            });
        return out;
    }


    template<typename R>
    Mat<R> Composite<R>::mul_add_mul_with_bias(std::initializer_list<Mat<R>> matrices) {
        vector<Mat<R>> matrices_vector(matrices);
        return mul_add_mul_with_bias(matrices_vector);
    }

    template<typename R>
    Mat<R> Composite<R>::mul_add_mul_with_bias(vector<Mat<R>>& matrices) {
        // broacast to largest input size
        dim_t max_broadcast = matrices[1].dims(1);
        for (auto matrices_ptr = matrices.begin()+1; matrices_ptr < matrices.end(); matrices_ptr+=2) {
            max_broadcast = std::max(max_broadcast, matrices_ptr->dims(1));
        }

        Mat<R> out(matrices[0].dims(0), max_broadcast, weights<R>::zeros());
        auto matrices_ptr = matrices.begin();
        while (matrices_ptr != (matrices.end() - 1)) {
            // inputs must either match the broadcasted size, or be broadcastable by having their
            // outer dimension be 1 (a column vector essentially)
            ASSERT2(((matrices_ptr+1)->dims(1) == max_broadcast) || ((matrices_ptr+1)->dims(0) == 1 && (matrices_ptr+1)->dims(1) == max_broadcast),
                "incompatible broadcast dimensions for mul_add_mul_with_bias");
            if ((matrices_ptr+1)->dims(1) == max_broadcast) {
                MAT(out) += dot(MAT(*matrices_ptr).wrapper(), MAT(*(matrices_ptr + 1)).wrapper());
            } else {
                TensorInternal<R, 2> temp(mshadow::Shape2(1, max_broadcast));
                temp = dot(MAT(*matrices_ptr).wrapper(), MAT(*(matrices_ptr + 1)).wrapper());
                MAT(out) += temp.wrapper()[0].template broadcast<0>(MAT(out).shape());
            }
            DEBUG_ASSERT_MAT_NOT_NAN(out)
            matrices_ptr+=2;
        }

        MAT(out) += MAT(matrices.back()).wrapper()[0].template broadcast<0>(MAT(out).shape());

        if (graph::backprop_enabled)
            graph::emplace_back([matrices, out, max_broadcast]() mutable {
                auto matrices_ptr = matrices.begin();
                while (matrices_ptr != (matrices.end() - 1)) {
                    if ((matrices_ptr+1)->dims(1) == max_broadcast) {
                        SAFE_GRAD(*matrices_ptr)     += dot(GRAD(out).wrapper(),              MAT(*(matrices_ptr+1)).wrapper().T());
                        SAFE_GRAD(*(matrices_ptr+1)) += dot(MAT(*matrices_ptr).wrapper().T(), GRAD(out).wrapper());
                    } else {
                        // broadcasting input means taking outer product here:
                        {
                            TensorInternal<R, 2> temp(mshadow::Shape2(1, max_broadcast));
                            temp.wrapper()[0] = sum_cols(GRAD(out).wrapper());
                            SAFE_GRAD(*matrices_ptr) += dot(
                                temp.wrapper(), MAT(*(matrices_ptr+1)).wrapper().T()
                            );
                        }
                        // broadcasting output means sum after the reverse product here:
                        {
                            TensorInternal<R, 2> temp(mshadow::Shape2((*matrices_ptr).dims(0), max_broadcast));
                            temp = dot(MAT(*matrices_ptr).wrapper().T(), GRAD(out).wrapper());
                            SAFE_GRAD(*(matrices_ptr+1)).wrapper()[0] += sum_cols(temp.wrapper());
                        }
                    }
                    matrices_ptr+=2;
                }
                SAFE_GRAD(matrices.back()).wrapper()[0] += sum_cols(GRAD(out).wrapper());
            });

        return out;
    }

    // operation of the form (A * x + B * y) + C, called with mul_add_mul_with_bias(A, x, B, y, C)
    template<typename R>
    Mat<R> Composite<R>::mul_add_mul_with_bias(
            Mat<R> matrix1,
            Mat<R> input_to_1,
            Mat<R> matrix2,
            Mat<R> input_to_2,
            Mat<R> bias) {
        ASSERT2(matrix1.dims(1) == input_to_1.dims(0), "matmul 1 dimensions misaligned.");
        ASSERT2(matrix2.dims(1) == input_to_2.dims(0), "matmul 2 dimensions misaligned.");
        ASSERT2(matrix2.dims(0) == bias.dims(1) && matrix1.dims(0) == bias.dims(1) && bias.dims(0) == 1,
            "Matrices cannot be computed in mul_add_mul_with_bias, they do not have the correct output dimensions.");

        if (input_to_1.dims(1) != input_to_2.dims(1)) {
            if (input_to_1.dims(1) == 1) {
                return mul_add_broadcast_mul_with_bias(matrix1, input_to_1, matrix2, input_to_2, bias);
            } else if (input_to_2.dims(1) == 1) {
                return mul_add_broadcast_mul_with_bias(matrix2, input_to_2, matrix1, input_to_1, bias);
            } else {
                ASSERT2(input_to_1.dims(1) == input_to_2.dims(1), "different output dimensions for inputs 1 and 2 to mul_add_mul_with_bias");
            }
        }

        Mat<R> out (matrix1.dims(0), input_to_1.dims(1), weights<R>::empty());

        MAT(out) = MAT(bias).wrapper()[0].template broadcast<0>(MAT(out).shape());
        MAT(out) += dot(MAT(matrix1).wrapper(), MAT(input_to_1).wrapper());
        MAT(out) += dot(MAT(matrix2).wrapper(), MAT(input_to_2).wrapper());

        if (graph::backprop_enabled)
            graph::emplace_back([matrix1, input_to_1, matrix2, input_to_2, bias, out]() mutable {
                // first multiply:
                // broadcasting input means taking outer product here:
                SAFE_GRAD(matrix1) += dot(GRAD(out).wrapper(), MAT(input_to_1).wrapper().T());
                // broadcasting output means sum after the reverse product here:
                SAFE_GRAD(input_to_1) += dot(MAT(matrix1).wrapper().T(), GRAD(out).wrapper());
                // broadcasting input means taking outer product here:
                SAFE_GRAD(matrix2) += dot(GRAD(out).wrapper(), MAT(input_to_2).wrapper().T());
                // broadcasting output means sum after the reverse product here:
                SAFE_GRAD(input_to_2) += dot(MAT(matrix2).wrapper().T(), GRAD(out).wrapper());
                // bias vector:
                SAFE_GRAD(bias).wrapper()[0] += sum_cols(GRAD(out).wrapper());
            });
        return out;
    }

    template class Composite<float>;
    template class Composite<double>;


}
