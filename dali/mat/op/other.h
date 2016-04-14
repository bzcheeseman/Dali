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

        static void grad(Mat<R>* mat);

        static Mat<R> consider_constant(Mat<R>);
        static Mat<R> consider_constant_if(Mat<R>, bool should_consider_constant);

        static bool is_nan(Mat<R> a);
        static bool is_grad_nan(Mat<R> a);
        static bool equals(Mat<R> a, Mat<R> b);
        static bool allclose(Mat<R> a, Mat<R> b, double tol);
        static bool grad_allclose(Mat<R> a, Mat<R> b, double tol);

        static std::vector<int> argsort(Mat<R>);
        static std::vector<size_t> argsort(const std::vector<Mat<R>>& mats);

        static int argmax(const Mat<R>& mat);
        static int argmin(const Mat<R>& mat);
        static std::vector<int> argmax(const Mat<R>& mat, int dimension);
        static std::vector<int> argmin(const Mat<R>& mat, int dimension);
        static int argmax_slice(const Mat<R>& mat, int lower, int upper);
        static int argmin_slice(const Mat<R>& mat, int lower, int upper);

        static void copy(Mat<R>* dest, const Mat<R>& source);
        static void copy_grad(Mat<R>* dest, const Mat<R>& source);
    };
}

#endif
