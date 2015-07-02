#include "dali/tensor/op/reshaping.h"

#include "dali/tensor/__MatMacros__.h"
#include "dali/math/TensorOps.h"
#include "dali/math/LazyTensor.h"

#define DONT_COMPILE

using std::vector;

namespace matops {

    template<typename R>
    Mat<R> Reshaping<R>::rows_pluck(
            Mat<R> matrix,
            Indexing::Index indices
            ) {
        #ifndef DONT_COMPILE
        Mat<R> out (
            matrix.dims(1),
            indices.size(),
            weights<R>::empty());

        for (std::size_t offset = 0; offset < indices.size(); ++offset) {
            MAT(out).col(offset) = MAT(matrix).row(indices[offset]).transpose();
        }
        if (graph::backprop_enabled) {
            graph::emplace_back([matrix, out, indices]() mutable {
                auto index_ptr = indices.data();
                for (std::size_t i = 0; i < out.dims(1); ++i) {
                    // for each row do the same operation as for row_pluck:
                    SAFE_GRAD(matrix).row(*index_ptr).noalias() += GRAD(out).col(i).transpose();
                    index_ptr++;
                }
            });
        }
        return out;
        #else
        return Mat<R>(1,1);
        #endif
    }


    template<typename R>
    Mat<R> Reshaping<R>::hstack(Mat<R> matrix1, Mat<R> matrix2) {
        return Reshaping<R>::hstack({matrix1, matrix2});
    }

    template<typename R>
    Mat<R> Reshaping<R>::hstack(std::initializer_list<Mat<R>> matrices) {
        vector<Mat<R>> matrices_vector(matrices);
        return hstack(matrices_vector);
    }

    template<typename R>
    Mat<R> Reshaping<R>::hstack(std::vector<Mat<R>>& matrices) {
        int n = -1;
        int d_total = 0;
        for (auto& mat : matrices) {
            if (n == -1) {
                n = mat.dims(0);
            } else {
                ASSERT2(mat.dims(0) == n, "Matrices cannot be joined -- they do not have the same number of rows.");
            }
            d_total+= mat.dims(1);
        }
        Mat<R> out (
            n, d_total, weights<R>::empty()
        );
        int offset = 0;
        int col, row;
        auto out_data = out.w().mutable_cpu_data();

        for (row = 0; row < n; row++) {
            offset = 0;
            for (auto& mat : matrices) {
                const int col_size = mat.dims(1);
                const auto mat_data = mat.w().cpu_data();
                for (col = 0; col < col_size; col++) {
                    *(out_data.dptr_ + (out_data.stride_ * row) + (col + offset)) = *(mat_data.dptr_ + (mat_data.stride_ * row) + col);
                }
                offset += col_size;
            }
        }

        if (graph::backprop_enabled)
            graph::emplace_back([matrices, out, n]() mutable {
                int offset = 0;
                auto out_data = out.dw().cpu_data();
                int row, col;
                for (row = 0; row < n; row++) {
                    offset = 0;
                    for (auto& mat : matrices) {
                        const int col_size = mat.dims(1);
                        auto mat_data = mat.dw().mutable_cpu_data();
                        for (col = 0; col < col_size; col++) {
                            *(mat_data.dptr_ + (mat_data.stride_ * row) + col ) += *(out_data.dptr_ + (out_data.stride_ * row) + (col + offset));
                        }
                        offset += col_size;
                    }
                }
            });
        return out;
    }

    template<typename R>
    Mat<R> Reshaping<R>::vstack(Mat<R> matrix1, Mat<R> matrix2) {
        return Reshaping<R>::vstack({matrix1, matrix2});
    }


    template<typename R>
    void Reshaping<R>::resize(const Mat<R>& mat, dim_t n, dim_t d) {
        #ifndef DONT_COMPILE
        mat.w()->dims[0] = n;
        mat.w()->dims[1] = d;
        MAT(mat).conservativeResize(n, d);
        GRAD(mat).conservativeResize(n, d);
        #else
        #endif
    }

    template<typename R>
    Mat<R> Reshaping<R>::vstack(std::initializer_list<Mat<R>> matrices) {
        vector<Mat<R>> matrices_vector(matrices);
        return vstack(matrices_vector);
    }

    template<typename R>
    Mat<R> Reshaping<R>::vstack(std::vector<Mat<R>>& matrices) {
        assert(matrices.size() > 0);
        int d = matrices[0].dims(1);
        int n_total = 0;
        for (auto& mat : matrices) {
            ASSERT2(mat.dims(1) == d,
                "Matrices cannot be vertically stacked -- "
                "they do not have the same number of cols.");
            n_total += mat.dims(0);
        }
        Mat<R> out (
            n_total,
            d,
            weights<R>::empty()
        );
        int offset = 0;
        for (auto& mat : matrices) {
            MAT(out).Slice(offset, offset + mat.dims(0)) = MAT(mat).wrapper() + (R)0.0;
            // MAT(out).mutable_cpu_data().Slice(offset, offset + mat.dims(0)) += MAT(mat).cpu_data();
            offset += mat.dims(0);
        }
        if (graph::backprop_enabled)
            graph::emplace_back([matrices, out]() mutable {
                int offset = 0;
                for (auto & mat : matrices) {
                    SAFE_GRAD(mat) +=
                            GRAD(out).Slice(offset, offset + mat.dims(0)).wrapper();
                    offset += mat.dims(0);
                }
            });
        return out;

    }


    template<typename R>
    Mat<R> Reshaping<R>::rows_cols_pluck(
            Mat<R> matrix,
            Indexing::Index row_indices,
            Indexing::Index col_indices) {
        #ifndef DONT_COMPILE
        ASSERT2(row_indices.size() != col_indices.size(),"Cannot pluck column row pairs, not the "
                "same amount of row and column indices.");
            Mat<R> out (
                1,
                row_indices.size(),
                weights<R>::empty());
            for (int offset = 0; offset < row_indices.size(); ++offset)
                MAT(out)(offset) = MAT(matrix)(row_indices[offset], col_indices[offset]);
        if (graph::backprop_enabled && !matrix.constant) {
            graph::emplace_back([matrix, out, row_indices, col_indices]() mutable {
                auto row_index_ptr = row_indices.data();
                auto col_index_ptr = col_indices.data();
                for (int i = 0; i < out.dims(1); ++i) {
                    // for each row do the same operatoitn as for row_pluck:
                    GRAD(matrix)(*row_index_ptr, *col_index_ptr) += GRAD(out)(i);
                    row_index_ptr++;
                    col_index_ptr++;
                }
            });
        }
        return out;
        #else
        return Mat<R>(1,1);
        #endif
    }

    template<typename R>
    Mat<R> Reshaping<R>::row_pluck(
            Mat<R> matrix,
            int row) {
        ASSERT2(
            row >= 0 && row < matrix.dims(0),
            utils::MS() << "Row must be positive and less than number of rows in matrix ("
                        << matrix.dims(0) << ")."
        );
        // performs a copy
        Mat<R> out(matrix.dims(1), 1, weights<R>::empty());
        MAT(out).ravel() = MAT(matrix)[row].wrapper();

        if (graph::backprop_enabled)
            graph::emplace_back([matrix, out, row]() mutable {
                SAFE_GRAD(matrix)[row] += GRAD(out).ravel().wrapper();
            });
        return out;
    }

    template<typename R>
    Mat<R> Reshaping<R>::col_pluck(
            Mat<R> matrix,
            int col) {
        #ifndef DONT_COMPILE
        Mat<R> out (1, matrix.dims(0), weights<R>::empty());
        MAT(out) = MAT(matrix).col(col);
        if (graph::backprop_enabled)
            graph::emplace_back([matrix, out, col]() mutable {
                SAFE_GRAD(matrix).col(col).noalias() += GRAD(out).col(0).transpose();
            });
        return out;
        #else
        return Mat<R>(1,1);
        #endif
    }


    template<typename R>
    Mat<R> Reshaping<R>::transpose(Mat<R> matrix) {
        Mat<R> out (
            matrix.dims(1),
            matrix.dims(0),
            weights<R>::empty());
        MAT(out) = MAT(matrix).wrapper().T();
        if (graph::backprop_enabled && !matrix.constant)
            graph::emplace_back([matrix, out]() mutable {
                GRAD(matrix) += (GRAD(out).wrapper()).T();
            });
        return out;
    }
    template class Reshaping<float>;
    template class Reshaping<double>;

}
