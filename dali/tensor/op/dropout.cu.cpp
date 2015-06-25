#include "dali/tensor/op/dropout.h"

#include "dali/tensor/__MatMacros__.h"
#include "dali/math/TensorOps.h"
#include "dali/math/LazyTensor.h"

using std::vector;

#define DONT_COMPILE

namespace matops {

    template<typename R>
    Mat<R> Dropout<R>::dropout(
            Mat<R> matrix,
            R drop_prob) {
        #ifndef DONT_COMPILE

        assert(0.0 <= drop_prob && drop_prob <= 1.0);

        // no dropout happens.
        if (drop_prob < 1e-6)
            return matrix;

        auto out = Mat<R>::empty_like(matrix);

        auto bool_mat = std::make_shared<Eigen::Matrix<R, Eigen::Dynamic, Eigen::Dynamic>>(
            matrix.dims(0),
            matrix.dims(1)
        );

        std::default_random_engine generator;
        std::bernoulli_distribution distribution(1.0 - drop_prob);
        std::random_device rd;
        generator.seed(rd());

        auto data_ptr = MAT(matrix).data();
        auto out_ptr  = MAT(out).data();
        auto bool_ptr = bool_mat->data();

        for (int i = 0; i < matrix.number_of_elements();++i) {
            (*bool_ptr) = distribution(generator) ? 1.0 : 0.0;
            (*out_ptr) = (*bool_ptr) > 0 ? *data_ptr : 0.0;
            out_ptr++;
            data_ptr++;
            bool_ptr++;
        }

        if (graph::backprop_enabled) {
            graph::emplace_back([matrix, out, bool_mat]() mutable {
                SAFE_GRAD(matrix) += (GRAD(out).array() * (*bool_mat).array()).matrix();
            });
        }
        return out;
        #else
        return Mat<R>(1,1);
        #endif
    }

    template<typename R>
    Mat<R> Dropout<R>::dropout_normalized(
            Mat<R> matrix,
            R drop_prob) {
        #ifndef DONT_COMPILE

        assert(0.0 <= drop_prob && drop_prob <= 1.0);

        // no dropout happens.
        if (drop_prob < 1e-6)
            return matrix;

        auto out = Mat<R>::empty_like(matrix);

        auto bool_mat = std::make_shared<Eigen::Matrix<R, Eigen::Dynamic, Eigen::Dynamic>>(
            matrix.dims(0),
            matrix.dims(1)
        );

        std::default_random_engine generator;
        std::bernoulli_distribution distribution(1.0 - drop_prob);
        std::random_device rd;
        generator.seed(rd());

        auto data_ptr = MAT(matrix).data();
        auto out_ptr  = MAT(out).data();
        auto bool_ptr = bool_mat->data();

        R normalized_drop_prob = 1.0 / (1.0 - drop_prob);
        for (unsigned int i = 0; i < matrix.number_of_elements();++i) {
            (*bool_ptr) = distribution(generator) ? normalized_drop_prob : 0.0;
            (*out_ptr) = (*bool_ptr) > 0 ? *data_ptr : 0.0;
            out_ptr++;
            data_ptr++;
            bool_ptr++;
        }

        if (graph::backprop_enabled) {
            graph::emplace_back([matrix, out, bool_mat]() mutable {
                SAFE_GRAD(matrix) += (GRAD(out).array() * (*bool_mat).array()).matrix();
            });
        }
        return out;
        #else
        return Mat<R>(1,1);
        #endif
    }

    template<typename R>
    vector<Mat<R>> Dropout<R>::dropout_normalized(
            const vector<Mat<R>>& matrices,
            R drop_prob) {
        #ifndef DONT_COMPILE
        vector<Mat<R>> dropped_matrices;
        dropped_matrices.reserve(matrices.size());
        for (auto& mat : matrices) {
            dropped_matrices.emplace_back(dropout_normalized(mat, drop_prob));
        }
        return dropped_matrices;
        #else
        return {Mat<R>(1,1)};
        #endif
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
    Mat<R> Dropout<R>::fast_dropout(Mat<R> matrix) {
        #ifndef DONT_COMPILE
        auto out = Mat<R>::empty_like(matrix);

        auto randn_mat = std::make_shared<Eigen::Matrix<R, Eigen::Dynamic, Eigen::Dynamic>>(
            matrix.dims(0),
            matrix.dims(1)
        );

        std::default_random_engine generator;
        std::normal_distribution<R> distribution(1.0, 1.0);
        std::random_device rd;
        generator.seed(rd());

        auto data_ptr = MAT(matrix).data();
        auto out_ptr  = MAT(out).data();
        auto randn_ptr = randn_mat->data();

        for (unsigned int i = 0; i < matrix.number_of_elements();++i) {
            (*randn_ptr) = distribution(generator);
            (*out_ptr) = (*randn_ptr) * *data_ptr;
            out_ptr++;
            data_ptr++;
            randn_ptr++;
        }

        if (graph::backprop_enabled) {
            graph::emplace_back([matrix, out, randn_mat]() mutable {
                SAFE_GRAD(matrix) += (GRAD(out).array() * (*randn_mat).array()).matrix();
            });
        }
        return out;
        #else
        return Mat<R>(1,1);
        #endif
    }

    template class Dropout<float>;
    template class Dropout<double>;

}
