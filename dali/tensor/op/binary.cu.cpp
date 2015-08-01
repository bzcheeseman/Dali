#include "dali/tensor/op/binary.h"

#include "dali/tensor/Mat.h"
#include "dali/tensor/__MatMacros__.h"
#include "dali/math/TensorOps.h"
#include "dali/math/LazyTensor.h"
#include "dali/utils/core_utils.h"

using utils::MS;
using std::vector;
using namespace TensorOps;

namespace matops {
    template<typename R>
    Mat<R> Binary<R>::eltmul_broadcast_colwise(
            Mat<R> matrix1,
            Mat<R> matrix2) {
        ASSERT2(matrix1.dims(0) == matrix2.dims(0) && matrix2.dims(1) == 1,
                MS() << "Matrices " << matrix1 << " and " << matrix2
                     << " cannot be element multiplied with broadcast,"
                     << " they do not have the same dimensions.");
        auto out = Mat<R>::empty_like(matrix1);
        MAT(out) = MAT(matrix1).wrapper() * MAT(matrix2).ravel().wrapper().template broadcast<0>(MAT(matrix1).shape);
        if (graph::backprop_enabled())
            graph::emplace_back([matrix1, matrix2, out]() mutable {
                SAFE_GRAD(matrix1) += GRAD(out).wrapper() * (MAT(matrix2).ravel().wrapper().template broadcast<0>(GRAD(out).shape));
                SAFE_GRAD(matrix2).ravel() += sum_cols(MAT(matrix1).wrapper() * GRAD(out).wrapper());
            });
        return out;
    }

    template<typename R>
    Mat<R> Binary<R>::eltdivide_broadcast(
            Mat<R> matrix1,
            Mat<R> matrix2) {
        ASSERT2(matrix1.dims(0) == matrix2.dims(0) && matrix2.dims(1) == 1,
                MS() << "Matrices " << matrix1 << " and " << matrix2
                     << " cannot be element divided with broadcast,"
                     << " they do not have the same dimensions.");
        auto out = Mat<R>::empty_like(matrix1);
        MAT(out) = (
            MAT(matrix1).wrapper()
            /
            MAT(matrix2).ravel().wrapper().template broadcast<0>(MAT(matrix1).shape)
        );
        if (graph::backprop_enabled())
            graph::emplace_back([matrix1, matrix2, out]() mutable {
                SAFE_GRAD(matrix1) += (
                    (GRAD(out)).wrapper() /
                    MAT(matrix2).ravel().wrapper().template broadcast<0>(GRAD(out).shape)
                );
                SAFE_GRAD(matrix2).ravel() -= sum_cols((
                    F<op::div_grad<R>>(
                        MAT(matrix1).wrapper(),
                        MAT(matrix2).ravel().wrapper().template broadcast<0>(MAT(matrix1).shape)
                    )
                ) * GRAD(out).wrapper());
            });
        return out;
    }

    template<typename R>
    Mat<R> Binary<R>::eltmul(
            Mat<R> matrix1,
            Mat<R> matrix2) {

        if (matrix1.dims(0) != matrix2.dims(0) && (matrix1.dims(0) == 1 || matrix2.dims(0) == 1)) {
            if (matrix1.dims(0) == 1) {
                return eltmul_broadcast_rowwise(matrix2, matrix1);
            }
            return eltmul_broadcast_rowwise(matrix1, matrix2);
        }

        ASSERT2(matrix1.dims(0) == matrix2.dims(0) && matrix1.dims(1) == matrix2.dims(1),
                "Matrices cannot be element-wise multiplied, they do not have the same dimensions.");
        auto out = Mat<R>::empty_like(matrix1);
        MAT(out) = MAT(matrix1).wrapper() * MAT(matrix2).wrapper();
        if (graph::backprop_enabled())
            graph::emplace_back([matrix1, matrix2, out]() mutable {
                SAFE_GRAD(matrix1) += MAT(matrix2).wrapper() * GRAD(out).wrapper();
                SAFE_GRAD(matrix2) += MAT(matrix1).wrapper() * GRAD(out).wrapper();
            });
        return out;
    }


    template<typename R>
    vector<Mat<R>> Binary<R>::eltmul(const vector<Mat<R>>& seq1, const vector<Mat<R>>& seq2) {
        ASSERT2(seq1.size() == seq2.size(), "Multiplying sequences of different sizes.");
        vector<Mat<R>> result(seq1.size());
        for (int i = 0; i < seq1.size(); ++i) {
            result[i] = seq1[i] * seq2[i];
        }
        return result;
    }


    template<typename R>
    Mat<R> Binary<R>::eltdivide(
            Mat<R> matrix1,
            Mat<R> matrix2) {
        if (matrix1.dims(1) != matrix2.dims(1) && (matrix1.dims(1) == 1 || matrix2.dims(1) == 1)) {
            if (matrix1.dims(1) == 1) {
                return eltdivide_broadcast_reversed(matrix2, matrix1);
            }
            return eltdivide_broadcast(matrix1, matrix2);
        }
        ASSERT2(matrix1.dims(0) == matrix2.dims(0) && matrix1.dims(1) == matrix2.dims(1),
                "Matrices cannot be element-wise divided, they do not have the same dimensions.");
        auto out = Mat<R>::empty_like(matrix1);
        MAT(out) = MAT(matrix1).wrapper() / MAT(matrix2).wrapper();
        if (graph::backprop_enabled())
            graph::emplace_back([matrix1, matrix2, out]() mutable {
                SAFE_GRAD(matrix1) += (
                    F<op::inv<R>>(MAT(matrix2).wrapper()) *
                    GRAD(out).wrapper()
                );
                SAFE_GRAD(matrix2) -= (
                    MAT(matrix1).wrapper() /
                    F<op::square<R>>(MAT(matrix2).wrapper())
                ) * GRAD(out).wrapper();
            });
        return out;
    }

    template<typename R>
    Mat<R> Binary<R>::add(
            Mat<R> matrix1,
            Mat<R> matrix2) {
        if (matrix1.dims(0) != matrix2.dims(0) && (matrix1.dims(0) == 1 || matrix2.dims(0) == 1)) {
            if (matrix1.dims(0) == 1) {
                // consider matrix1 to be a vector
                return add_broadcast_rowwise(matrix2, matrix1);
            }
            // consider matrix2 to be a vector
            return add_broadcast_rowwise(matrix1, matrix2);
        }
        ASSERT2(matrix1.dims() == matrix2.dims(), "Matrices cannot be added, they do not have the same dimensions.");

        auto out = Mat<R>::empty_like(matrix1);
        MAT(out) = MAT(matrix1).wrapper() + MAT(matrix2).wrapper();

        if (graph::backprop_enabled())
            graph::emplace_back([matrix1, matrix2, out]() mutable {
                SAFE_GRAD(matrix1) += GRAD(out).wrapper();
                SAFE_GRAD(matrix2) += GRAD(out).wrapper();
            });
        return out;
    }


    template<typename R>
    Mat<R> Binary<R>::sub(
            Mat<R> matrix1,
            Mat<R> matrix2) {
        if (matrix1.dims(1) != matrix2.dims(1) && (matrix1.dims(1) == 1 || matrix2.dims(1) == 1)) {
            if (matrix1.dims(1) == 1) {
                // consider matrix1 to be a vector
                return sub_broadcast_reversed(matrix2, matrix1);
            }
            // consider matrix2 to be a vector
            return sub_broadcast(matrix1, matrix2);
        }

        ASSERT2(matrix1.dims() == matrix2.dims(), "Matrices cannot be subtracted, they do not have the same dimensions.");

        auto out = Mat<R>::empty_like(matrix1);
        MAT(out) = MAT(matrix1).wrapper() - MAT(matrix2).wrapper();

        if (graph::backprop_enabled())
            graph::emplace_back([matrix1, matrix2, out]() mutable {
                SAFE_GRAD(matrix1) += GRAD(out).wrapper();
                SAFE_GRAD(matrix2) -= GRAD(out).wrapper();
            });

        return out;
    }

    template<typename R>
    Mat<R> Binary<R>::add_broadcast_rowwise(Mat<R> matrix1, Mat<R> matrix2) {
        // broadcast matrix 2:
        ASSERT2(matrix2.dims(0) == 1, "Second argument to add_broadcast must be a row vector (first dimension=1)");
        ASSERT2(matrix1.dims(1) == matrix2.dims(1),
                MS() << "vector-like argument to add_broadcast_rowwise must have length (" << matrix2.dims(1)
                     << ") equal to outer dimension of first argument (" << matrix1.dims(1) << ").");
        auto out = Mat<R>::empty_like(matrix1);
        MAT(out) = (
            MAT(matrix1).wrapper() +
            MAT(matrix2).ravel().wrapper().template broadcast<1>(
                MAT(matrix1).shape
            )
        );
        if (graph::backprop_enabled())
            graph::emplace_back([matrix1, matrix2, out]() mutable {
                SAFE_GRAD(matrix1) += GRAD(out).wrapper();
                SAFE_GRAD(matrix2).ravel() += sum_rows(GRAD(out).wrapper());
            });
        return out;
    }

    template<typename R>
    Mat<R> Binary<R>::add_broadcast_colwise(Mat<R> matrix1, Mat<R> matrix2) {
        // broadcast matrix 2:
        ASSERT2(matrix2.dims(1) == 1, "Second argument to add_broadcast must be a col vector (second dimension=1)");
        ASSERT2(matrix1.dims(0) == matrix2.dims(0),
                MS() << "vector-like argument to add_broadcast_colwise must have outer dimension (" << matrix2.dims(0)
                     << ") equal to inner dimension of first argument (" << matrix1.dims(0) << ").");
        auto out = Mat<R>::empty_like(matrix1);
        MAT(out) = (
            MAT(matrix1).wrapper() +
            MAT(matrix2).ravel().wrapper().template broadcast<0>(
                MAT(matrix1).shape
            )
        );
        if (graph::backprop_enabled())
            graph::emplace_back([matrix1, matrix2, out]() mutable {
                SAFE_GRAD(matrix1) += GRAD(out).wrapper();
                SAFE_GRAD(matrix2).ravel() += sum_cols(GRAD(out).wrapper());
            });
        return out;
    }

    template<typename R>
    Mat<R> Binary<R>::sub_broadcast(Mat<R> matrix1, Mat<R> matrix2) {
        // broadcast matrix 2:
        ASSERT2(matrix2.dims(1) == 1, "Second argument to sub_broadcast must be a vector (second dimension=1)");
        if (matrix1.dims(0) != matrix2.dims(0)) {
            ASSERT2(matrix1.dims(0) == matrix2.dims(0),
                MS() << "vector-like argument to sub_broadcast must have outer dimension (" << matrix2.dims(0)
                     << ") equal to inner dimension of first argument (" << matrix1.dims(0) << ").");
        }
        auto out = Mat<R>::empty_like(matrix1);
        MAT(out) = (
            MAT(matrix1).wrapper() -
            MAT(matrix2).ravel().wrapper().template broadcast<0>(
                MAT(matrix1).shape
            )
        );
        if (graph::backprop_enabled())
            graph::emplace_back([matrix1, matrix2, out]() mutable {
                SAFE_GRAD(matrix1) += GRAD(out).wrapper();
                SAFE_GRAD(matrix2).ravel() -= sum_cols(GRAD(out).wrapper());
            });
        return out;
    }

    template<typename R>
    Mat<R> Binary<R>::sub_broadcast_reversed(Mat<R> matrix1, Mat<R> matrix2) {
        // broadcast matrix 2:
        ASSERT2(matrix2.dims(1) == 1, "Second argument to sub_broadcast_reversed must be a vector (first dimension=1)");
        if (matrix1.dims(0) != matrix2.dims(0)) {
            ASSERT2(matrix1.dims(0) == matrix2.dims(0),
                MS() << "vector-like argument to sub_broadcast_reversed must have outer dimension (" << matrix2.dims(0)
                     << ") equal to inner dimension of first argument (" << matrix1.dims(0) << ").");
        }
        auto out = Mat<R>::empty_like(matrix1);
        MAT(out) = (
            MAT(matrix2).ravel().wrapper().template broadcast<0>(
                MAT(matrix1).shape
            ) - MAT(matrix1).wrapper()
        );
        if (graph::backprop_enabled())
            graph::emplace_back([matrix1, matrix2, out]() mutable {
                SAFE_GRAD(matrix1) -= GRAD(out).wrapper();
                SAFE_GRAD(matrix2).ravel() += sum_cols(GRAD(out).wrapper());
            });
        return out;
    }

    // not GPU friendly.
    template<typename R>
    Mat<R> Binary<R>::pow(Mat<R> matrix, Mat<R> other) {
        ASSERT2(other.dims(0) == 1 && other.dims(1) == 1, "exponent must be a 1x1 matrix.");
        auto out = Mat<R>::empty_like(matrix);
        // TODO (szymon): it would be better it was done completely on GPU.
        R exponent_val = MAT(other)(0);
        MAT(out) = F<op::power<R>>(MAT(matrix).wrapper(), exponent_val);
        if (graph::backprop_enabled())
            graph::emplace_back([matrix, out, other, exponent_val]() mutable {
                SAFE_GRAD(matrix) += exponent_val * F<op::power<R>>(MAT(matrix).wrapper(), exponent_val - (R)1.0) * GRAD(out).wrapper();
                if (!other.constant) {
                    TensorInternal<R,2> temp(MAT(matrix).shape);
                    temp = F<op::log_or_zero<R>>(MAT(matrix).wrapper()) * MAT(out).wrapper() * GRAD(out).wrapper();
                    GRAD(other) += temp.sum();
                }
            });
        return out;
    }

    template<typename R>
    vector<Mat<R>> Binary<R>::eltmul_broadcast_rowwise(const vector<Mat<R>>& seq1, const vector<Mat<R>>& seq2) {
        ASSERT2(seq1.size() == seq2.size(), "Multiplying sequences of different sizes.");

        vector<Mat<R>> result(seq1.size());
        for (int i = 0; i < seq1.size(); ++i) {
            result[i] = eltmul_broadcast_rowwise(seq1[i], seq2[i]);
        }
        return result;
    }

    template<typename R>
    vector<Mat<R>> Binary<R>::eltmul_broadcast_colwise(const vector<Mat<R>>& seq1, const vector<Mat<R>>& seq2) {
        ASSERT2(seq1.size() == seq2.size(), "Multiplying sequences of different sizes.");

        vector<Mat<R>> result(seq1.size());
        for (int i = 0; i < seq1.size(); ++i) {
            result[i] = eltmul_broadcast_colwise(seq1[i], seq2[i]);
        }
        return result;
    }

    template<typename R>
    vector<Mat<R>> Binary<R>::eltmul_rowwise(const vector<Mat<R>>& seq1, const vector<Mat<R>>& seq2) {
        ASSERT2(seq1.size() == seq2.size(), "Multiplying sequences of different sizes.");

        vector<Mat<R>> result(seq1.size());
        for (int i = 0; i < seq1.size(); ++i) {
            result[i] = eltmul_rowwise(seq1[i], seq2[i]);
        }
        return result;
    }



    template<typename R>
    Mat<R> Binary<R>::add(std::initializer_list<Mat<R>> matrices) {
        auto matrices_vector = vector<Mat<R>>(matrices);
        return add(matrices_vector);
    }

    template<typename R>
    Mat<R> Binary<R>::add(std::vector<Mat<R>>& matrices) {
        ASSERT2(matrices.size() > 0, "Got 0 matrices to add.");

        auto out = Mat<R>::zeros_like(matrices.front());
        for (auto& matrix : matrices)
            MAT(out) += MAT(matrix).wrapper();
        if (graph::backprop_enabled())
            graph::emplace_back([matrices, out]() mutable {
                for (auto& matrix : matrices) {
                    SAFE_GRAD(matrix) += GRAD(out).wrapper();
                }
            });

        return out;
    }

    template<typename R>
    Mat<R> Binary<R>::mul(
            Mat<R> matrix1,
            Mat<R> matrix2) {
        ASSERT2(matrix1.dims(1) == matrix2.dims(0), "matrix product dimensions misaligned.");
        Mat<R> out (matrix1.dims(0), matrix2.dims(1), weights<R>::empty());

        MAT(out) = dot( MAT(matrix1).wrapper(), MAT(matrix2).wrapper() );

        if (graph::backprop_enabled())
            graph::emplace_back([matrix1, matrix2, out]() mutable {

                SAFE_GRAD(matrix1) += dot( GRAD(out).wrapper(),        MAT(matrix2).wrapper().T() );
                SAFE_GRAD(matrix2) += dot( MAT(matrix1).wrapper().T(), GRAD(out).wrapper() );
            });
        return out;
    }

    template<typename R>
    Mat<R> Binary<R>::eltdivide_broadcast_reversed(
            Mat<R> matrix1,
            Mat<R> matrix2) {
        ASSERT2(matrix1.dims(0) == matrix2.dims(0) && matrix2.dims(1) == 1,
                MS() << "Matrices " << matrix1 << " and " << matrix2
                     << " cannot be element divided with broadcast,"
                     << " they do not have the same dimensions.");
        auto out = Mat<R>::empty_like(matrix1);
        MAT(out) = (
            MAT(matrix2).ravel().wrapper().template broadcast<0>(MAT(matrix1).shape)
            /
            MAT(matrix1).wrapper()
        );
        if (graph::backprop_enabled())
            graph::emplace_back([matrix1, matrix2, out]() mutable {
                SAFE_GRAD(matrix1) -= F<op::div_grad<R>>(
                    MAT(matrix2).ravel().wrapper().template broadcast<0>(MAT(matrix1).shape),
                    MAT(matrix1).wrapper()
                ) * GRAD(out).wrapper();
                SAFE_GRAD(matrix2).ravel() += sum_cols(
                    GRAD(out).wrapper() / MAT(matrix1).wrapper()
                );
            });
        return out;
    }

    template<typename R>
    Mat<R> Binary<R>::eltmul_broadcast_rowwise(
            Mat<R> matrix1,
            Mat<R> row_vector) {
        ASSERT2(matrix1.dims(1) == row_vector.dims(1) && row_vector.dims(0) == 1,
            "Matrices A and B^T cannot be element multiplied with broadcast, they do not have the same dimensions.");
        auto out = Mat<R>::empty_like(matrix1);
        MAT(out) = MAT(matrix1).wrapper() * MAT(row_vector).ravel().wrapper().template broadcast<1>(MAT(matrix1).shape);
        if (graph::backprop_enabled())
            graph::emplace_back([matrix1, row_vector, out]() mutable {
                SAFE_GRAD(matrix1) += GRAD(out).wrapper() * MAT(row_vector).ravel().wrapper().template broadcast<1>(GRAD(out).shape);
                SAFE_GRAD(row_vector).ravel() += sum_rows(MAT(matrix1).wrapper() * GRAD(out).wrapper());
            });
        return out;
    }

    template<typename R>
    Mat<R> Binary<R>::eltmul_rowwise(
        Mat<R> matrix1,
        Mat<R> matrix2) {

        ASSERT2(matrix1.dims(0) == matrix2.dims(1) && matrix1.dims(1) == matrix2.dims(0),
            "Matrices A and B^T cannot be element-wise multiplied, they do not have the same dimensions.");
        auto out = Mat<R>::empty_like(matrix1);
        MAT(out) = MAT(matrix1).wrapper() * MAT(matrix2).wrapper().T();
        if (graph::backprop_enabled())
            graph::emplace_back([matrix1, matrix2, out]() mutable {
                SAFE_GRAD(matrix1) += (
                    MAT(matrix2).wrapper().T() * GRAD(out).wrapper()
                );
                SAFE_GRAD(matrix2) += (
                    MAT(matrix1).wrapper().T() *
                    GRAD(out).wrapper().T()
                );
            });
        return out;
    }

    template class Binary<float>;
    template class Binary<double>;
    template class Binary<int>;
}
