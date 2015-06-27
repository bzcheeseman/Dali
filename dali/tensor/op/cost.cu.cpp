#include "dali/tensor/op/cost.h"

#include "dali/tensor/__MatMacros__.h"
#include "dali/math/TensorOps.h"
#include "dali/math/LazyTensor.h"

#define DONT_COMPILE

using std::vector;

namespace Indexing {
    typedef uint ind_t;
    class Index {
        ind_t& operator[](std::size_t idx);
        ind_t  operator[](std::size_t idx) const;
    };
}

namespace matops {
    template<typename R>
    Mat<R> Cost<R>::softmax_no_grad(Mat<R> matrix, R temperature) {
        ASSERT2(temperature == 1.0, "Not implemented yet (Temperature != 1.0 for softmax).");
        auto out = Mat<R>::empty_like(matrix);
        MAT(out) = MAT(matrix).wrapper().softmax();
        return out;
    }

    template<typename R>
    Mat<R> Cost<R>::softmax(Mat<R> matrix, R temperature) {
        #ifndef DONT_COMPILE
        Mat<R> out = Cost<R>::softmax_no_grad(matrix, temperature);
        if (graph::backprop_enabled && !matrix.constant)
            graph::emplace_back([matrix, temperature, out]() mutable {
                auto& dw = GRAD(matrix);
                auto& sm = MAT(out);
                auto& dy = GRAD(out);
                eigen_mat sm_times_dy = (sm.array() * dy.array());
                auto colwise_sums                      = sm_times_dy.colwise().sum();
                for (size_t i = 0; i < matrix.dims(1); ++i) {
                    dw.col(i) += (sm_times_dy.col(i) - sm.col(i) * colwise_sums(i)) / temperature;
                }
            });
        return out;
        #else
        return Mat<R>(1,1);
        #endif
    }

    template<typename R>
    Mat<R> Cost<R>::softmax_transpose(Mat<R> matrix, R temperature) {
        #ifndef DONT_COMPILE
        Mat<R> out = Cost<R>::softmax_no_grad_transpose(matrix, temperature);
        if (graph::backprop_enabled && !matrix.constant)
            graph::emplace_back([matrix, temperature, out]() mutable {
                auto& dw = GRAD(matrix);
                auto& sm = MAT(out);
                auto& dy = GRAD(out);
                eigen_mat sm_times_dy = (sm.array() * dy.array());
                auto rowwise_sums                      = sm_times_dy.rowwise().sum();
                for (size_t i = 0; i < matrix.dims(0); ++i) {
                    dw.row(i) += sm_times_dy.row(i) - sm.row(i) * rowwise_sums(i);
                }
            });
        return out;
        #else
        return Mat<R>(1,1);
        #endif
    }

    template<typename R>
    Mat<R> Cost<R>::softmax_no_grad_transpose(Mat<R> matrix, R temperature) {
        #ifndef DONT_COMPILE
        auto out = Mat<R>::empty_like(matrix);
        auto layer_max = MAT(matrix).rowwise().maxCoeff().array().matrix();
        auto exped_distributions = (MAT(matrix).colwise() - layer_max.row(0)).array().exp().matrix();

        auto total_distribution = exped_distributions.rowwise().sum().array().matrix();
        MAT(out) = (exped_distributions.array().colwise() / total_distribution.col(0).array());
        return out;
        #else
        return Mat<R>(1,1);
        #endif
    }

    template<typename R>
    vector<Mat<R>> Cost<R>::softmax_no_grad(const vector<Mat<R>>& matrices, R temperature) {
        vector<Mat<R>> out;
        out.reserve(matrices.size());
        ASSERT2(matrices.size() > 0, "Must be a non empty list of vectors to softmax.");
        R layer_max = matrices.front().w(0);
        for (auto& mat : matrices) {
            ASSERT2(mat.dims(0) == 1 && mat.dims(1) == 1, "Softmax on a vector must be made on 1x1 matrices only.");
            layer_max = std::max(layer_max, mat.w(0));
        }
        R total = 0.0;
        for (auto& mat : matrices) {
            out.emplace_back(1,1);
            out.back().w(0) = std::exp(mat.w(0) - layer_max) / temperature;
            total += out.back().w(0);
        }
        for (auto& mat : out) {
            mat.w(0) /= total;
        }
        return out;
    }

    template<typename R>
    vector<Mat<R>> Cost<R>::softmax(vector<Mat<R>>& matrices, R temperature) {
        vector<Mat<R>> out = Cost<R>::softmax_no_grad(matrices, temperature);
        if (graph::backprop_enabled)
            graph::emplace_back([temperature, out, matrices]() mutable {
                R colwise_sums = 0.0;

                for (int i = 0; i < out.size(); i++) {
                    colwise_sums += out[i].w(0) * out[i].dw(0);
                }

                for (int i = 0; i < out.size(); i++) {
                    if (!matrices[i].constant) {
                        matrices[i].dw(0) += (
                            out[i].w(0) * out[i].dw(0) - out[i].w(0) * colwise_sums
                        ) / temperature;

                        // dw.col(i) += (sm_times_dy.col(i) - sm.col(i) * colwise_sums(i)) / temperature;
                    }
                }
            });
        return out;
    }


    template<typename R>
    Mat<R> Cost<R>::sigmoid_binary_cross_entropy(Mat<R> matrix, R t) {
        #ifndef DONT_COMPILE
        assert(0 <= t && t <= 1);
        assert(matrix.dims().size() > 1);
        auto out = Mat<R>::empty_like(matrix);

        auto sigmoided_input = std::make_shared<eigen_mat>(
            MAT(matrix).array().unaryExpr(utils::sigmoid_operator<R>())
        );

        MAT(out) = -(
                              t  * ( sigmoided_input->array()   + EPS      ).log()
                    + ( 1.0 - t) * ( 1.00000001 - sigmoided_input->array() ).log()
        ).matrix();

        if (graph::backprop_enabled())
            graph::emplace_back([matrix, t, out, sigmoided_input]() mutable {
                SAFE_GRAD(matrix).array() += (sigmoided_input->array() - t) * GRAD(out).array();
            });
        return out;
        #else
        return Mat<R>(1,1);
        #endif
    }

    template<typename R>
    Mat<R> Cost<R>::binary_cross_entropy(Mat<R> matrix, R t) {
        #ifndef DONT_COMPILE
        assert(0 <= t && t <= 1);
        assert(matrix.dims().size() > 1);
        Mat<R> out =  Mat<R>(
            matrix.dims(0),
            matrix.dims(1),
            weights<R>::empty());

        auto x = MAT(matrix).array();

        MAT(out) = (-(t * (x + EPS).log() + (1.0-t) * (1.0 - x + EPS).log())).matrix();

        DEBUG_ASSERT_MAT_NOT_NAN(out);

        if (graph::backprop_enabled())
            graph::emplace_back([matrix, t, out]() mutable {
                auto x = MAT(matrix).array();
                SAFE_GRAD(matrix).array() += (
                    (
                        (t - x) /
                        (x * (x - 1.0) + EPS)
                    ) * GRAD(out).array()
                );
                DEBUG_ASSERT_GRAD_NOT_NAN(matrix);
            });

        return out;
        #else
        return Mat<R>(1,1);
        #endif
    }

    template<typename R>
    Mat<R> Cost<R>::cross_entropy(Mat<R> matrix, uint answer_idx) {
        #ifndef DONT_COMPILE
        DEBUG_ASSERT_BOUNDS(MAT(matrix),0.0,1.0 + EPS);
        assert(matrix.dims().size() > 1);
        assert(answer_idx < matrix.dims(0));
        Mat<R> out =  Mat<R>(1, matrix.dims(1), weights<R>::empty());

        auto x = MAT(matrix).array();
        MAT(out) = - (x.row(answer_idx).array() + EPS).log();

        DEBUG_ASSERT_MAT_NOT_NAN(out);

        if (graph::backprop_enabled)
            graph::emplace_back([matrix, answer_idx, out]() mutable {
                auto x = MAT(matrix).array();
                SAFE_GRAD(matrix).row(answer_idx).array() += -(x.row(answer_idx).array() + EPS).inverse() * GRAD(out).array();
            });
        return out;
        #else
        return Mat<R>(1,1);
        #endif
    }

    template<typename R>
    Mat<R> Cost<R>::cross_entropy(Mat<R> matrix, Mat<R> target) {
        #ifndef DONT_COMPILE
        ASSERT2(matrix.dims(0) == target.dims(0) && matrix.dims(1) == target.dims(1),
            "Matrix and target must have same dimension");

        Mat<R> out = Mat<R>::empty_like(matrix);
        MAT(out) = -(MAT(target).array() * ((MAT(matrix).array() + EPS).log())).matrix();

        DEBUG_ASSERT_NOT_NAN(MAT(out));

        if (graph::backprop_enabled)
            graph::emplace_back([matrix, target, out]() mutable {
                auto x = MAT(matrix).array();
                SAFE_GRAD(matrix).noalias() -= (((x + EPS).inverse()) * MAT(target).array() * GRAD(out).array()).matrix();
                SAFE_GRAD(target).noalias() -= ((x.log()) * GRAD(out).array()).matrix();
            });
        return out;
        #else
        return Mat<R>(1,1);
        #endif
    }


    template<typename R>
    Mat<R> Cost<R>::softmax_cross_entropy(Mat<R> matrix, uint answer_idx) {
        #ifndef DONT_COMPILE
        Mat<R> out =  Mat<R>(1, 1, weights<R>::empty());
        Mat<R> probs = softmax_no_grad(matrix);
        MAT(out)(0,0) = -std::log(MAT(probs)(answer_idx, 0));

        if (graph::backprop_enabled)
            graph::emplace_back([matrix, probs, answer_idx, out]() mutable {
                SAFE_GRAD(matrix) += MAT(probs) * GRAD(out)(0,0);
                // write gradients into log probabilities
                SAFE_GRAD(matrix)(answer_idx, 0) -= 1 * GRAD(out)(0,0);
            });
        return out;
        #else
        return Mat<R>(1,1);
        #endif
    }

    template<typename R>
    Mat<R> Cost<R>::softmax_cross_entropy(Mat<R> matrix, Indexing::Index targets) {
        #ifndef DONT_COMPILE
        Mat<R> out =  Mat<R>(1, targets.size(), weights<R>::empty());
        Mat<R> probs = softmax_no_grad(matrix);
        for (int i = 0; i < targets.size(); i++) {
            MAT(out)(i) = -std::log(MAT(probs)(targets[i], i));
        }

        if (graph::backprop_enabled)
            graph::emplace_back([matrix, probs, out, targets]() mutable {
                if (!matrix.constant) {
                    SAFE_GRAD(matrix).noalias() += (MAT(probs).array().rowwise() * GRAD(out).row(0).array()).matrix();
                    for (int i = 0; i < targets.size(); i++) {
                        GRAD(matrix)(targets[i],i) -= 1.0 * GRAD(out)(i);
                    }
                }
            });
        return out;
        #else
        return Mat<R>(1,1);
        #endif
    }

    template<typename R>
    Mat<R> Cost<R>::margin_loss(Mat<R> matrix, uint answer_idx, R margin) {
        #ifndef DONT_COMPILE
        // Exprected input is a column vector
        assert(answer_idx < matrix.dims(0));
        assert(matrix.dims(1) == 1);
        Mat<R> error(1,1);
        for (int idx = 0; idx < matrix.dims(0); ++idx) {
            if (idx == answer_idx) continue;
            error = error + MatOps<R>::max(matrix[idx] - matrix[answer_idx] + margin, 0.0);
        }
        return error;
        #else
        return Mat<R>(1,1);
        #endif
    }

    template class Cost<float>;
    template class Cost<double>;


}
