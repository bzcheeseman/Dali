#ifndef DALI_TENSOR_OP_RESHAPING_H
#define DALI_TENSOR_OP_RESHAPING_H

#include "dali/tensor/Mat.h"
#include "dali/tensor/Tape.h"
#include "dali/utils.h"

template<typename R> class Mat;

namespace matops {
    template<typename R>
    struct Reshaping {
        static Mat<R> hstack(Mat<R>, Mat<R>);
        static Mat<R> hstack(std::initializer_list<Mat<R>>);
        static Mat<R> hstack(std::vector<Mat<R>>&);
        static Mat<R> vstack(Mat<R>, Mat<R>);
        static Mat<R> vstack(std::initializer_list<Mat<R>>);
        static Mat<R> vstack(std::vector<Mat<R>>&);
        static Mat<R> transpose(Mat<R>);
        static Mat<R> rows_pluck(Mat<R>, Indexing::Index);
        static Mat<R> rows_cols_pluck(Mat<R>, Indexing::Index, Indexing::Index);
        static Mat<R> row_pluck(Mat<R>, int);
        static Mat<R> col_pluck(Mat<R>, int);
        static void resize(const Mat<R>& mat, dim_t rows, dim_t cols);
    };
}

#endif
