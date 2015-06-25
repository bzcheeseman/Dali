#ifndef DALI_TENSOR_OP_OTHER_H
#define DALI_TENSOR_OP_OTHER_H

#include "dali/tensor/Mat.h"
#include "dali/tensor/Tape.h"
#include "dali/utils.h"

template<typename R> class Mat;

namespace matops {
    template<typename R>
    struct Other {
        static Mat<R> fill(Mat<R>, R);

        static Mat<R> consider_constant(Mat<R>);
        static Mat<R> consider_constant_if(Mat<R>, bool should_consider_constant);

        static bool equals(Mat<R> a, Mat<R> b);
        static bool allclose(Mat<R> a, Mat<R> b, R tol);
        static bool grad_allclose(Mat<R> a, Mat<R> b, R tol);

        static std::vector<size_t> argsort_rowwise(Mat<R>);
        static std::vector<size_t> argsort(const std::vector<Mat<R>>& mats);

        static void resize(const Mat<R>& mat, dim_t rows, dim_t cols);
        static int argmax(const Mat<R>& mat);
        static int argmin(const Mat<R>& mat);
        static std::vector<int> argmax(const Mat<R>& mat, int dimension);
        static std::vector<int> argmin(const Mat<R>& mat, int dimension);
        static int argmax_slice(const Mat<R>& mat, int lower, int upper);
    };
}

#endif
