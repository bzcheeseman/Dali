#include "dali/tensor/op/dropout.h"

#include "dali/tensor/__MatMacros__.h"
#include "dali/math/TensorOps.h"
#include "dali/math/LazyTensor.h"

using std::vector;
using std::make_shared;

#define DONT_COMPILE

namespace matops {

    template<typename R>
    Mat<R> Dropout<R>::dropout(
            Mat<R> matrix,
            R drop_prob) {

        assert(0.0 <= drop_prob && drop_prob <= 1.0);

        // no dropout happens.
        if (drop_prob < 1e-6)
            return matrix;

        auto out = Mat<R>::empty_like(matrix);

        auto mask = make_shared<TensorInternal<R, 2>>(MAT(matrix).shape());
        weights<R>::bernoulli(1.0 - drop_prob)(*mask);

        MAT(out) = MAT(matrix).wrapper() * (*mask).wrapper();

        if (graph::backprop_enabled) {
            graph::emplace_back([matrix, out, mask]() mutable {
                SAFE_GRAD(matrix) += (GRAD(out).wrapper() * (*mask).wrapper());
            });
        }
        return out;
    }

    template<typename R>
    Mat<R> Dropout<R>::dropout_normalized(
            Mat<R> matrix,
            R drop_prob) {

        assert(0.0 <= drop_prob && drop_prob <= 1.0);

        // no dropout happens.
        if (drop_prob < 1e-6)
            return matrix;

        auto out = Mat<R>::empty_like(matrix);

        auto mask = make_shared<TensorInternal<R, 2>>(MAT(matrix).shape());
        weights<R>::bernoulli_normalized(1.0 - drop_prob)(*mask);

        MAT(out) = MAT(matrix).wrapper() * (*mask).wrapper();

        if (graph::backprop_enabled) {
            graph::emplace_back([matrix, out, mask]() mutable {
                SAFE_GRAD(matrix) += (GRAD(out).wrapper() * (*mask).wrapper());
            });
        }
        return out;
    }

    template<typename R>
    vector<Mat<R>> Dropout<R>::dropout_normalized(
            const vector<Mat<R>>& matrices,
            R drop_prob) {
        vector<Mat<R>> dropped_matrices;
        dropped_matrices.reserve(matrices.size());
        for (auto& mat : matrices) {
            dropped_matrices.emplace_back(dropout_normalized(mat, drop_prob));
        }
        return dropped_matrices;
    }

    template<typename R>
    vector<Mat<R>> Dropout<R>::dropout(
            const vector<Mat<R>>& matrices,
            R drop_prob) {
        vector<Mat<R>> dropped_matrices;
        dropped_matrices.reserve(matrices.size());
        for (auto& mat : matrices) {
            dropped_matrices.emplace_back(dropout(mat, drop_prob));
        }
        return dropped_matrices;
    }

    template<typename R>
    Mat<R> Dropout<R>::fast_dropout(
            Mat<R> matrix) {

        auto out = Mat<R>::empty_like(matrix);

        auto mask = make_shared<TensorInternal<R, 2>>(MAT(matrix).shape());
        weights<R>::gaussian(1.0, 1.0)(*mask);

        MAT(out) = MAT(matrix).wrapper() * (*mask).wrapper();

        if (graph::backprop_enabled) {
            graph::emplace_back([matrix, out, mask]() mutable {
                SAFE_GRAD(matrix) += (GRAD(out).wrapper() * (*mask).wrapper());
            });
        }
        return out;
    }

    template class Dropout<float>;
    template class Dropout<double>;

}
