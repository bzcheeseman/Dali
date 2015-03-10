#include "Softmax.h"

template<typename T>
Mat<T> softmax_transpose(Mat<T> matrix) {

        DEBUG_ASSERT_NOT_NAN(matrix.w);

        auto layer_max = matrix.w.rowwise().maxCoeff().array().matrix();
        auto exped_distributions = (matrix.w.colwise() - layer_max.col(0)).array().exp().matrix();

        auto out = Mat<T>(
                matrix.dims(0),
                matrix.dims(1),
                false);

        auto total_distribution = exped_distributions.rowwise().sum().array().matrix();
        out.w = (exped_distributions.array().colwise() / total_distribution.col(0).array());

        DEBUG_ASSERT_POSITIVE(out.w);

        return out;
}

template<typename T>
Mat<T> softmax(const Mat<T> matrix) {

        DEBUG_ASSERT_NOT_NAN(matrix.w);

        auto layer_max = matrix.w.colwise().maxCoeff().array().matrix();
        auto exped_distributions = (matrix.w.rowwise() - layer_max.row(0)).array().exp().matrix();

        auto out = Mat<T>(
                matrix.dims(0),
                matrix.dims(1),
                false);

        auto total_distribution = exped_distributions.colwise().sum().array().matrix();
        out.w = (exped_distributions.array().rowwise() / total_distribution.row(0).array());

        DEBUG_ASSERT_POSITIVE(out.w);

        return out;
}

template Mat<float> softmax(Mat<float>);
template Mat<double> softmax(Mat<double>);

template Mat<float> softmax_transpose(Mat<float>);
template Mat<double> softmax_transpose(Mat<double>);
