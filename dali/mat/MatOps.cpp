#include "MatOps.h"

using std::vector;
using std::string;
using utils::assert2;
using utils::MS;
using utils::LambdaOperator;

#define GRAD(X) if (!(X).constant) (X).dw()

template<typename R>
Mat<R> MatOps<R>::eltmul_broadcast(
        Mat<R> matrix1,
        Mat<R> matrix2) {
    assert2(matrix1.dims(0) == matrix2.dims(0) && matrix2.dims(1) == 1,
            MS() << "Matrices " << matrix1 << " and " << matrix2
                 << " cannot be element multiplied with broadcast,"
                 << " they do not have the same dimensions.");
    auto out = Mat<R>::empty_like(matrix1);
    out.w() = (matrix1.w().array().colwise() * matrix2.w().col(0).array()).matrix();
    if (graph::backprop_enabled)
        graph::emplace_back([matrix1, matrix2, out]() {
            GRAD(matrix1).noalias() += ((out.dw()).array().colwise() * (matrix2.w()).col(0).array()).matrix();
            GRAD(matrix2).noalias() += ((matrix1.w()).array() * (out.dw()).array()).matrix().rowwise().sum();
        });
    return out;
}

template<typename R>
Mat<R> MatOps<R>::eltdivide_broadcast(
        Mat<R> matrix1,
        Mat<R> matrix2) {
    assert2(matrix1.dims(0) == matrix2.dims(0) && matrix2.dims(1) == 1,
            MS() << "Matrices " << matrix1 << " and " << matrix2
                 << " cannot be element divided with broadcast,"
                 << " they do not have the same dimensions.");
    auto out = Mat<R>::empty_like(matrix1);
    out.w() = (matrix1.w().array().colwise() / matrix2.w().col(0).array()).matrix();
    if (graph::backprop_enabled)
        graph::emplace_back([matrix1, matrix2, out]() {
            GRAD(matrix1).noalias() += ((out.dw()).array().colwise() * (matrix2.w()).col(0).array().inverse()).matrix();
            GRAD(matrix2).noalias() -= (
                (
                    matrix1.w().array().colwise() /
                    matrix2.w().col(0).array().square()
                ).matrix().array() * out.dw().array()
            ).matrix().rowwise().sum();
        });
    return out;
}

template<typename R>
Mat<R> MatOps<R>::eltdivide_broadcast_reversed(
        Mat<R> matrix1,
        Mat<R> matrix2) {
    assert2(matrix1.dims(0) == matrix2.dims(0) && matrix2.dims(1) == 1,
            MS() << "Matrices " << matrix1 << " and " << matrix2
                 << " cannot be element divided with broadcast,"
                 << " they do not have the same dimensions.");
    auto out = Mat<R>::empty_like(matrix1);
    out.w() = (matrix1.w().array().inverse().colwise() * matrix2.w().col(0).array()).matrix();
    if (graph::backprop_enabled)
        graph::emplace_back([matrix1, matrix2, out]() {
            GRAD(matrix1).noalias() -= ((matrix1.w().array().square().inverse().colwise() * matrix2.w().col(0).array()).matrix().array() * out.dw().array()).matrix();
            GRAD(matrix2).noalias() += (matrix1.w().array().inverse() * out.dw().array()).rowwise().sum().matrix();
        });
    return out;
}

template<typename R>
Mat<R> MatOps<R>::eltmul(
    Mat<R> matrix1,
    Mat<R> matrix2) {
    if (matrix1.dims(1) != matrix2.dims(1) && (matrix1.dims(1) == 1 || matrix2.dims(1) == 1)) {
        if (matrix1.dims(1) == 1) {
            return eltmul_broadcast(matrix2, matrix1);
        }
        return eltmul_broadcast(matrix1, matrix2);
    }
    assert2(matrix1.dims(0) == matrix2.dims(0) && matrix1.dims(1) == matrix2.dims(1),
            "Matrices cannot be element-wise multiplied, they do not have the same dimensions.");
    auto out = Mat<R>::empty_like(matrix1);
    out.w() = (matrix1.w().array() * matrix2.w().array()).matrix();
    if (graph::backprop_enabled)
        graph::emplace_back([matrix1, matrix2, out]() {
            GRAD(matrix1).noalias() += ((matrix2.w()).array() * (out.dw()).array()).matrix();
            GRAD(matrix2).noalias() += ((matrix1.w()).array() * (out.dw()).array()).matrix();
        });
    return out;
}

template<typename R>
Mat<R> MatOps<R>::max(Mat<R> matrix, R lower_bound) {
    auto out = Mat<R>::empty_like(matrix);
    // out = max(matrix, lower_bound);
    out.w() = matrix.w().unaryExpr(
        LambdaOperator<R>([&lower_bound](R item) {
            return std::max(item, lower_bound);
        }));
    if (graph::backprop_enabled)
        graph::emplace_back([matrix, out, lower_bound]() {
            if (!matrix.constant) {
                // mask = (matrix >= lower_bound) ? 1.0 : 0:0;

                auto mask = matrix.w().unaryExpr(
                    LambdaOperator<R>([&lower_bound](R item) {
                        return item >= lower_bound ? 1.0 : 0.0;
                    }));
                matrix.dw().noalias() += (mask.array() * (out.dw()).array()).matrix();
            }
        });
    return out;
}

template<typename R>
Mat<R> MatOps<R>::eltmul(
    Mat<R> matrix,
    R alpha) {

    auto out = Mat<R>::empty_like(matrix);
    out.w() = (matrix.w().array() * alpha).matrix();
    if (graph::backprop_enabled)
        graph::emplace_back([matrix, alpha, out]() {
            GRAD(matrix).noalias() += (alpha * (out.dw()).array()).matrix();
        });
    return out;
}

template<typename R>
vector<Mat<R>> MatOps<R>::eltmul(const vector<Mat<R>>& seq1, const vector<Mat<R>>& seq2) {
    assert2(seq1.size() == seq2.size(), "Multiplying sequences of different sizes.");

    vector<Mat<R>> result(seq1.size());
    for (int i = 0; i < seq1.size(); ++i) {
        result[i] = seq1[i] * seq2[i];
    }
    return result;
}


template<typename R>
Mat<R> MatOps<R>::eltdivide(
    Mat<R> matrix1,
    Mat<R> matrix2) {
    if (matrix1.dims(1) != matrix2.dims(1) && (matrix1.dims(1) == 1 || matrix2.dims(1) == 1)) {
        if (matrix1.dims(1) == 1) {
            return eltdivide_broadcast_reversed(matrix2, matrix1);
        }
        return eltdivide_broadcast(matrix1, matrix2);
    }
    assert2(matrix1.dims(0) == matrix2.dims(0) && matrix1.dims(1) == matrix2.dims(1),
            "Matrices cannot be element-wise divided, they do not have the same dimensions.");
    auto out = Mat<R>::empty_like(matrix1);
    out.w() = (matrix1.w().array() / matrix2.w().array()).matrix();
    if (graph::backprop_enabled)
        graph::emplace_back([matrix1, matrix2, out]() {
            GRAD(matrix1).noalias() += ((matrix2.w()).array().inverse() * (out.dw()).array()).matrix();
            GRAD(matrix2).noalias() -= (((matrix1.w()).array() / matrix2.w().array().square()) * (out.dw()).array()).matrix();
        });
    return out;
}


template<typename R>
Mat<R> MatOps<R>::eltdivide(
    Mat<R> matrix,
    R alpha) {

    auto out = Mat<R>::empty_like(matrix);
    out.w() = (matrix.w().array() / alpha).matrix();
    if (graph::backprop_enabled)
        graph::emplace_back([matrix, alpha, out]() {
            GRAD(matrix).noalias() += ((1.0 / alpha) * (out.dw()).array()).matrix();
        });
    return out;
}


template<typename R>
Mat<R> MatOps<R>::eltmul_broadcast_rowwise(
    Mat<R> matrix1,
    Mat<R> row_vector) {
    if (matrix1.dims(1) != row_vector.dims(1) || row_vector.dims(0) != 1)
        throw std::invalid_argument("Matrices A and B^T cannot be element multiplied with broadcast, they do not have the same dimensions.");
    auto out = Mat<R>::empty_like(matrix1);
    out.w() = (matrix1.w().array().rowwise() * row_vector.w().row(0).array()).matrix();
    if (graph::backprop_enabled)
        graph::emplace_back([matrix1, row_vector, out]() {
            GRAD(matrix1).noalias() += ((out.dw()).array().rowwise() * (row_vector.w()).row(0).array()).matrix();
            GRAD(row_vector).noalias() += (((matrix1.w()).array() * (out.dw()).array()).matrix().colwise().sum()).matrix();
        });
    return out;
}

template<typename R>
vector<Mat<R>> MatOps<R>::eltmul_broadcast_rowwise(const vector<Mat<R>>& seq1, const vector<Mat<R>>& seq2) {
    assert2(seq1.size() == seq2.size(), "Multiplying sequences of different sizes.");

    vector<Mat<R>> result(seq1.size());
    for (int i = 0; i < seq1.size(); ++i) {
        result[i] = eltmul_broadcast_rowwise(seq1[i], seq2[i]);
    }
    return result;
}

template<typename R>
Mat<R> MatOps<R>::eltmul_rowwise(
    Mat<R> matrix1,
    Mat<R> matrix2) {

    if (matrix1.dims(0) != matrix2.dims(1) || matrix1.dims(1) != matrix2.dims(0))
        throw std::invalid_argument("Matrices A and B^T cannot be element-wise multiplied, they do not have the same dimensions.");
    auto out = Mat<R>::empty_like(matrix1);
    out.w() = (matrix1.w().array() * matrix2.w().transpose().array()).matrix();
    if (graph::backprop_enabled)
        graph::emplace_back([matrix1, matrix2, out]() {
            GRAD(matrix1).noalias() += ((matrix2.w()).transpose().array() * (out.dw()).array()).matrix();
            GRAD(matrix2).noalias() += ((matrix1.w()).array() * (out.dw()).array()).matrix().transpose();
        });
    return out;
}

template<typename R>
vector<Mat<R>> MatOps<R>::eltmul_rowwise(const vector<Mat<R>>& seq1, const vector<Mat<R>>& seq2) {
    assert2(seq1.size() == seq2.size(), "Multiplying sequences of different sizes.");

    vector<Mat<R>> result(seq1.size());
    for (int i = 0; i < seq1.size(); ++i) {
        result[i] = eltmul_rowwise(seq1[i], seq2[i]);
    }
    return result;
}

template<typename R>
Mat<R> MatOps<R>::add(
        Mat<R> matrix1,
        Mat<R> matrix2) {
    if (matrix1.dims(1) != matrix2.dims(1) && (matrix1.dims(1) == 1 || matrix2.dims(1) == 1)) {
        if (matrix1.dims(1) == 1) {
            return add_broadcast(matrix2, matrix1);
        }
        return add_broadcast(matrix1, matrix2);
    }
    assert2((matrix1.dims(0) == matrix2.dims(0)) && (matrix1.dims(1) == matrix2.dims(1)),
        "Matrices cannot be added, they do not have the same dimensions.");
    auto out = Mat<R>::empty_like(matrix1);
    out.w() = matrix1.w() + matrix2.w();
    if (graph::backprop_enabled)
        graph::emplace_back([matrix1, matrix2, out]() {
            GRAD(matrix1).noalias() += out.dw();
            GRAD(matrix2).noalias() += out.dw();
        });
    return out;
}


template<typename R>
Mat<R> MatOps<R>::sub(
        Mat<R> matrix1,
        Mat<R> matrix2) {
    if (matrix1.dims(1) != matrix2.dims(1) && (matrix1.dims(1) == 1 || matrix2.dims(1) == 1)) {
        if (matrix1.dims(1) == 1) {
            return sub_broadcast_reversed(matrix2, matrix1);
        }
        return sub_broadcast(matrix1, matrix2);
    }
    assert2((matrix1.dims(0) == matrix2.dims(0)) && (matrix1.dims(1) == matrix2.dims(1)),
        "Matrices cannot be subtracted, they do not have the same dimensions.");
    auto out = Mat<R>::empty_like(matrix1);
    out.w() = matrix1.w() - matrix2.w();
    if (graph::backprop_enabled)
        graph::emplace_back([matrix1, matrix2, out]() {
            GRAD(matrix1).noalias() += out.dw();
            GRAD(matrix2).noalias() -= out.dw();
        });
    return out;
}

template<typename R>
Mat<R> MatOps<R>::add(
        Mat<R> matrix1,
        R alpha) {
    auto out = Mat<R>::empty_like(matrix1);
    out.w().array() = matrix1.w().array() + alpha;
    if (graph::backprop_enabled)
        graph::emplace_back([matrix1, out]() {
            GRAD(matrix1).noalias() += out.dw();
        });
    return out;
}

template<typename R>
Mat<R> MatOps<R>::add_broadcast(Mat<R> matrix1, Mat<R> matrix2) {
    // broadcast matrix 2:
    if (matrix1.dims(0) != matrix2.dims(0) || matrix2.dims(1) != 1)
            throw std::invalid_argument("Matrices cannot be added with broadcast, they do not have the same dimensions.");
    auto out = Mat<R>::empty_like(matrix1);
    out.w() = (matrix1.w().colwise() + matrix2.w().col(0)).matrix();
    if (graph::backprop_enabled)
        graph::emplace_back([matrix1, matrix2, out]() {
            GRAD(matrix1).noalias() += out.dw();
            GRAD(matrix2).noalias() += out.dw().rowwise().sum();
        });
    return out;
}

template<typename R>
Mat<R> MatOps<R>::sub_broadcast(Mat<R> matrix1, Mat<R> matrix2) {
    // broadcast matrix 2:
    if (matrix1.dims(0) != matrix2.dims(0) || matrix2.dims(1) != 1)
        throw std::invalid_argument("Matrices cannot be substracted with broadcast, they do not have the same dimensions.");
    auto out = Mat<R>::empty_like(matrix1);
    out.w() = (matrix1.w().colwise() - matrix2.w().col(0)).matrix();
    if (graph::backprop_enabled)
        graph::emplace_back([matrix1, matrix2, out]() {
            GRAD(matrix1).noalias() += out.dw();
            GRAD(matrix2).noalias() -= out.dw().rowwise().sum();
        });
    return out;
}

template<typename R>
Mat<R> MatOps<R>::sub_broadcast_reversed(Mat<R> matrix1, Mat<R> matrix2) {
    // broadcast matrix 2:
    if (matrix1.dims(0) != matrix2.dims(0) || matrix2.dims(1) != 1)
        throw std::invalid_argument("Matrices cannot be substracted with broadcast, they do not have the same dimensions.");
    auto out = Mat<R>::empty_like(matrix1);
    out.w() = ((-matrix1.w()).colwise() + matrix2.w().col(0)).matrix();
    if (graph::backprop_enabled)
        graph::emplace_back([matrix1, matrix2, out] () {
            GRAD(matrix1).noalias() -= out.dw();
            GRAD(matrix2).noalias() += out.dw().rowwise().sum();
        });
    return out;
}

template<typename R>
Mat<R> MatOps<R>::sub_broadcast_reversed(Mat<R> matrix, R other) {
    auto out = Mat<R>::empty_like(matrix);
    out.w() = (other - matrix.w().array()).matrix();
    if (graph::backprop_enabled)
        graph::emplace_back([matrix, out] () {
            GRAD(matrix).noalias() -= out.dw();
        });
    return out;
}

template<typename R>
Mat<R> MatOps<R>::add(std::initializer_list<Mat<R>> matrices) {
    auto matrices_vector = vector<Mat<R>>(matrices);
    return add(matrices_vector);
}

template<typename R>
Mat<R> MatOps<R>::add(const std::vector<Mat<R>>& matrices) {
    auto out = Mat<R>::zeros_like(*matrices.begin());
    for (auto& matrix : matrices)
        out.w() += matrix.w();
    if (graph::backprop_enabled)
        graph::emplace_back([matrices, out]() {
            for (auto& matrix : matrices) {
                GRAD(matrix).noalias() += out.dw();
            }
        });
    return out;
}

template<typename R>
Mat<R> MatOps<R>::square(Mat<R> matrix) {
    auto out = Mat<R>::empty_like(matrix);
    out.w() = matrix.w().array().square();
    if (graph::backprop_enabled)
        graph::emplace_back([matrix, out]() {
            GRAD(matrix).noalias() += 2.0 * ((matrix.w()).array() * (out.dw()).array()).matrix();
        });
    return out;
}

template<typename R>
Mat<R> MatOps<R>::L2_norm(Mat<R> matrix) {
    auto out = Mat<R>(1, 1, false);
    out.w()(0) = matrix.w().norm();
    if (graph::backprop_enabled)
        graph::emplace_back([matrix, out]() {
            GRAD(matrix).noalias() += ((matrix.w().array() / out.w()(0)) * out.dw()(0)).matrix();
        });
    return out;
}

template<typename R>
Mat<R> MatOps<R>::sqrt(Mat<R> matrix) {
    auto out = Mat<R>::empty_like(matrix);
    out.w() = matrix.w().array().sqrt();
    if (graph::backprop_enabled)
        graph::emplace_back([matrix, out]() {
            GRAD(matrix).noalias() += 0.5 * ((out.w()).array().inverse() * (out.dw()).array()).matrix();
        });
    return out;
}

template<typename R>
Mat<R> MatOps<R>::elt_inv(Mat<R> matrix) {
    auto out = Mat<R>::empty_like(matrix);
    out.w() = matrix.w().array().inverse();
    if (graph::backprop_enabled)
        graph::emplace_back([matrix, out]() {
            GRAD(matrix).noalias() += -((out.w()).array().square() * (out.dw()).array()).matrix();
        });
    return out;
}

template<typename R>
Mat<R> MatOps<R>::fill(Mat<R> matrix, R filler) {
    auto out = Mat<R>::empty_like(matrix);
    out.w().fill(filler);
    return out;
}

template<typename R>
Mat<R> MatOps<R>::pow(Mat<R> matrix, R other) {
    if (other == (R) -1.0) {
        return MatOps<R>::elt_inv(matrix);
    } else if (other == (R) 0.0){
        return MatOps<R>::fill(matrix, 1.0);
    } else if (other == (R)0.5) {
        return MatOps<R>::sqrt(matrix);
    } else if (other == (R)1.0) {
        return matrix;
    } else if (other == (R)2.0) {
        return MatOps<R>::square(matrix);
    }
    auto out = Mat<R>::empty_like(matrix);
    out.w() = matrix.w().array().pow(other);
    if (graph::backprop_enabled)
        graph::emplace_back([matrix, out, other]() {
            GRAD(matrix).noalias() += other * ((matrix.w()).array().pow(other - 1.0) * (out.dw()).array()).matrix();
        });
    return out;
}

template<typename R>
Mat<R> MatOps<R>::pow(Mat<R> matrix, Mat<R> other) {
    assert2(other.dims(0) == 1 && other.dims(1) == 1, "exponent must be a 1x1 matrix.");
    auto out = Mat<R>::empty_like(matrix);
    out.w() = matrix.w().array().pow(other.w()(0));
    if (graph::backprop_enabled)
        graph::emplace_back([matrix, out, other]() {
            GRAD(matrix).noalias() += other.w()(0) * ((matrix.w()).array().pow(other.w()(0) - 1.0) * (out.dw()).array()).matrix();
            GRAD(other)(0) += (
                matrix.w().unaryExpr(utils::log_or_zero<R>()).array() * out.w().array() * out.dw().array()
            ).sum();
        });
    return out;
}

template<typename R>
Mat<R> MatOps<R>::sigmoid(Mat<R> matrix) {
    auto out = Mat<R>::empty_like(matrix);
    out.w() = matrix.w().unaryExpr(utils::sigmoid_operator<R>());
    if (graph::backprop_enabled)
        graph::emplace_back([matrix, out](){
            GRAD(matrix).noalias() += (((out.w()).array() - out.w().array().square()) * out.dw().array()).matrix();
        });
    return out;
}


template<typename R>
Mat<R> MatOps<R>::softmax_no_grad(Mat<R> matrix, R temperature) {
    auto out = Mat<R>::empty_like(matrix);
    auto layer_max = matrix.w().colwise().maxCoeff().array().matrix();
    auto exped_distributions = (matrix.w().rowwise() - layer_max.row(0)).array().exp().matrix();

    auto total_distribution = exped_distributions.colwise().sum().array().matrix();
    out.w() = (exped_distributions.array().rowwise() / total_distribution.row(0).array());
    return out;
}

template<typename R>
Mat<R> MatOps<R>::softmax(Mat<R> matrix, R temperature) {
    Mat<R> out = MatOps<R>::softmax_no_grad(matrix, temperature);
    if (graph::backprop_enabled && !matrix.constant)
        graph::emplace_back([matrix, temperature, out]() {
            auto& dw = matrix.dw();
            auto& sm = out.w();
            auto& dy = out.dw();
            typename Mat<R>::eigen_mat sm_times_dy = (sm.array() * dy.array());
            auto colwise_sums                      = sm_times_dy.colwise().sum();
            for (size_t i = 0; i < matrix.dims(1); ++i) {
                dw.col(i) += sm_times_dy.col(i) - sm.col(i) * colwise_sums(i);
            }
        });
    return out;
}

template<typename R>
vector<Mat<R>> MatOps<R>::softmax_no_grad(const vector<Mat<R>>& matrices, R temperature) {
    vector<Mat<R>> out;
    out.reserve(matrices.size());
    assert2(matrices.size() > 0, "Must be a non empty list of vectors to softmax.");
    R layer_max = matrices[0].w()(0);

    for (auto& mat : matrices) {
        assert2(mat.dims(0) == 1 && mat.dims(1) == 1, "Softmax on a vector must be made on 1x1 matrices only.");
        layer_max = std::max(layer_max, mat.w()(0));
    }
    R total = 0.0;
    for (auto& mat : matrices) {
        out.emplace_back(1,1);
        out.back().w()(0) = std::exp(mat.w()(0) - layer_max);
        total += out.back().w()(0);
    }
    for (auto& mat : out) {
        mat.w()(0) /= total;
    }
    return out;
}

template<typename R>
vector<Mat<R>> MatOps<R>::softmax(const vector<Mat<R>>& matrices, R temperature) {
    vector<Mat<R>> out = MatOps<R>::softmax_no_grad(matrices, temperature);
    if (graph::backprop_enabled)
        graph::emplace_back([temperature, out, matrices]() {
            R colwise_sums = 0.0;

            for (int i = 0; i < out.size(); i++) {
                auto& dw = matrices[i].dw();
                auto& sm = out[i].w();
                auto& dy = out[i].dw();
                colwise_sums += sm(0) * dy(0);
            }

            for (int i = 0; i < out.size(); i++) {
                if (!matrices[i].constant) {
                    auto& dw = matrices[i].dw();
                    auto& sm = out[i].w();
                    auto& dy = out[i].dw();
                    dw(0) += (sm(0) * dy(0)) - (sm(0) * colwise_sums);
                }
            }
        });
    return out;
}

template<typename R>
Mat<R> MatOps<R>::steep_sigmoid(Mat<R> matrix, R aggressiveness) {
    auto out = Mat<R>::empty_like(matrix);
    out.w() = matrix.w().unaryExpr(utils::steep_sigmoid_operator<R>(aggressiveness));
    if (graph::backprop_enabled)
        graph::emplace_back([matrix, out, aggressiveness](){
            GRAD(matrix).noalias() += (aggressiveness * ((out.w()).array() - out.w().array().square()) * out.dw().array()).matrix();
        });
    return out;
}

template<typename R>
Mat<R> MatOps<R>::sum(Mat<R> matrix) {
    Mat<R> out (1,1, false);
    out.w()(0) = matrix.w().array().sum();
    if (graph::backprop_enabled)
        graph::emplace_back([matrix, out]() {
            GRAD(matrix).array() += out.dw()(0);
        });
    return out;
}


template<typename R>
Mat<R> MatOps<R>::mean(Mat<R> matrix) {
    Mat<R> out (1,1, false);
    out.w()(0) = matrix.w().array().mean();
    if (graph::backprop_enabled)
        graph::emplace_back([matrix, out](){
            GRAD(matrix).array() += (1.0 / (matrix.number_of_elements())) * out.dw()(0);
        });
    return out;
}



template<typename R>
Mat<R> MatOps<R>::sigmoid_binary_cross_entropy(Mat<R> matrix, R t) {
    assert(0 <= t && t <= 1);
    assert(matrix.dims().size() > 1);
    auto out = Mat<R>::empty_like(matrix);

    auto sigmoided_input = std::make_shared<typename Mat<R>::eigen_mat>(
        matrix.w().array().unaryExpr(utils::sigmoid_operator<R>())
    );

    out.w() = -(
                          t  * ( sigmoided_input->array()   + EPS      ).log()
                + ( 1.0 - t) * ( 1.00000001 - sigmoided_input->array() ).log()
    ).matrix();

    if (graph::backprop_enabled)
        graph::emplace_back([matrix, t, out, sigmoided_input](){
            GRAD(matrix).array() += (sigmoided_input->array() - t) * out.dw().array();
        });
    return out;
}

template<typename R>
Mat<R> MatOps<R>::binary_cross_entropy(Mat<R> matrix, R t) {
    assert(0 <= t && t <= 1);
    assert(matrix.dims().size() > 1);
    Mat<R> out =  Mat<R>(
        matrix.dims(0),
        matrix.dims(1),
        false);

    auto x = matrix.w().array();

    out.w() = (-(t * (x + EPS).log() + (1.0-t) * (1.0 - x + EPS).log())).matrix();

    DEBUG_ASSERT_NOT_NAN(out.w());

    if (graph::backprop_enabled)
        graph::emplace_back([matrix, t, out](){
            auto x = matrix.w().array();
            GRAD(matrix).array() += (t-x) / (x*(x- 1.0) + EPS)* out.dw().array();
            DEBUG_ASSERT_NOT_NAN(matrix.dw());
        });

    return out;
}

template<typename R>
Mat<R> MatOps<R>::cross_entropy(Mat<R> matrix, uint answer_idx) {
    DEBUG_ASSERT_BOUNDS(matrix.w(),0.0,1.0 + EPS);
    assert(matrix.dims().size() > 1);
    assert(answer_idx < matrix.dims(0));
    Mat<R> out =  Mat<R>(
        1,
        matrix.dims(1),
        false);

    auto x = matrix.w().array();
    out.w() = - (x.row(answer_idx).array() + EPS).log();

    DEBUG_ASSERT_NOT_NAN(out.w());

    if (graph::backprop_enabled)
        graph::emplace_back([matrix, answer_idx, out](){
            auto x = matrix.w().array();
            GRAD(matrix).row(answer_idx).array() += -(x.row(answer_idx).array() + EPS).inverse() * out.dw().array();
        });
    return out;
}


template<typename R>
Mat<R> MatOps<R>::softmax_cross_entropy(Mat<R> matrix, uint answer_idx) {
    Mat<R> out =  Mat<R>(1, 1, false);

    Mat<R> probs = softmax(matrix);
    out.w()(0,0) = -std::log(probs.w()(answer_idx, 0));

    if (graph::backprop_enabled)
        graph::emplace_back([matrix, probs, answer_idx, out](){
            GRAD(matrix) += probs.w()* out.dw()(0,0);
            // write gradients into log probabilities
            GRAD(matrix)(answer_idx, 0) -= 1 * out.dw()(0,0);
        });
    return out;
}

template<typename R>
Mat<R> MatOps<R>::margin_loss(Mat<R> matrix, uint answer_idx, R margin) {
    // Exprected input is a column vector
    assert(answer_idx < matrix.dims(0));
    assert(matrix.dims(1) == 1);
    Mat<R> error(1,1);
    for (int idx = 0; idx < matrix.dims(0); ++idx) {
        if (idx == answer_idx) continue;
        error = error + MatOps<R>::max(matrix[idx] - matrix[answer_idx] + margin, 0.0);
    }
    return error;
}

template<typename R>
Mat<R> MatOps<R>::log(Mat<R> matrix) {
    assert(matrix.dims().size() > 1);
    auto out = Mat<R>::empty_like(matrix);
    out.w() = matrix.w().array().log();
    if (graph::backprop_enabled)
        graph::emplace_back([matrix, out](){
            GRAD(matrix).noalias() += ((1.0 / (matrix.w()).array()) * (out.dw()).array()).matrix();
        });
    return out;
}

template<typename R>
Mat<R> MatOps<R>::exp(Mat<R> matrix) {
    assert(matrix.dims().size() > 1);
    auto out = Mat<R>::empty_like(matrix);
    out.w() = matrix.w().array().exp();
    if (graph::backprop_enabled)
        graph::emplace_back([matrix, out](){
            GRAD(matrix).noalias() += ((out.w()).array() * (out.dw()).array()).matrix();
        });
    return out;
}

template<typename R>
Mat<R> MatOps<R>::hstack(Mat<R> matrix1, Mat<R> matrix2) {
    if (matrix1.dims(0) != matrix2.dims(0))
        throw std::invalid_argument("Matrices cannot be joined -- they do not have the same number of rows.");
    Mat<R> out (
        matrix1.dims(0),
        matrix1.dims(1) + matrix2.dims(1),
        false
    );
    out.w().block(0,0, matrix1.dims(0), matrix1.dims(1)) = matrix1.w();
    out.w().block(0,matrix1.dims(1), matrix2.dims(0), matrix2.dims(1)) = matrix2.w();
    if (graph::backprop_enabled)
        graph::emplace_back([matrix1, matrix2, out]() {
            GRAD(matrix1).noalias() += out.dw().block(0,0, matrix1.dims(0), matrix1.dims(1));
            GRAD(matrix2).noalias() += out.dw().block(0,matrix1.dims(1), matrix2.dims(0), matrix2.dims(1));
        });
    return out;
}

template<typename R>
Mat<R> MatOps<R>::hstack(std::initializer_list<Mat<R>> matrices) {
    vector<Mat<R>> matrices_vector(matrices);
    return hstack(matrices_vector);
}

template<typename R>
Mat<R> MatOps<R>::hstack(const std::vector<Mat<R>>& matrices) {
    int n = -1;
    int d_total = 0;
    for (auto& mat : matrices) {
        if (n == -1) {
            n = mat.dims(0);
        } else {
            if (mat.dims(0) != n) {
                throw std::invalid_argument("Matrices cannot be joined -- they do not have the same number of rows.");
            }
        }
        d_total+= mat.dims(1);
    }
    Mat<R> out (
        n,
        d_total,
        false
    );
    int offset = 0;
    for (auto& mat : matrices) {
        out.w().block(0, offset, mat.dims(0), mat.dims(1)) = mat.w();
        offset += mat.dims(1);
    }
    if (graph::backprop_enabled)
        graph::emplace_back([matrices, out]() {
            int offset = 0;
            for (auto & mat : matrices) {
                GRAD(mat).noalias() += out.dw().block(0, offset, mat.dims(0), mat.dims(1));
                offset += mat.dims(1);
            }
        });
    return out;
}

template<typename R>
Mat<R> MatOps<R>::vstack(Mat<R> matrix1, Mat<R> matrix2) {
    if (matrix1.dims(1) != matrix2.dims(1))
        throw std::invalid_argument("Matrices cannot be horizontally stacked -- they do not have the same number of cols.");
    Mat<R> out (
        matrix1.dims(0) + matrix2.dims(0),
        matrix1.dims(1),
        false
    );
    out.w().block(0,0, matrix1.dims(0), matrix1.dims(1)) = matrix1.w();
    out.w().block(matrix1.dims(0),0, matrix2.dims(0), matrix2.dims(1)) = matrix2.w();
    if (graph::backprop_enabled)
        graph::emplace_back([matrix1, matrix2, out]() {
            GRAD(matrix1).noalias() += out.dw().block(0,0, matrix1.dims(0), matrix1.dims(1));
            GRAD(matrix2).noalias() += out.dw().block(matrix1.dims(0),0, matrix2.dims(0), matrix2.dims(1));
        });
    return out;
}

template<typename R>
Mat<R> MatOps<R>::vstack(std::initializer_list<Mat<R>> matrices) {
    vector<Mat<R>> matrices_vector(matrices);
    return vstack(matrices_vector);
}

template<typename R>
Mat<R> MatOps<R>::vstack(const std::vector<Mat<R>>& matrices) {
    assert(matrices.size() > 0);
    assert(matrices[0].dims().size() > 1);
    int d = matrices[0].dims(1);
    int n_total = 0;
    for (auto& mat : matrices) {
        if (mat.dims(1) != d) {
            throw std::invalid_argument("Matrices cannot be horizontally stacked -- they do not have the same number of cols.");
        }
        n_total += mat.dims(0);
    }
    Mat<R> out (
        n_total,
        d,
        false
    );
    int offset = 0;
    for (auto& mat : matrices) {
        out.w().block(offset, 0, mat.dims(0), mat.dims(1)) = mat.w();
        offset += mat.dims(0);
    }
    if (graph::backprop_enabled)
        graph::emplace_back([matrices, out]() {
            int offset = 0;
            for (auto & mat : matrices) {
                GRAD(mat).noalias() += out.dw().block(offset,0, mat.dims(0), mat.dims(1));
                offset += mat.dims(0);
            }
        });
    return out;
}

template<typename R>
Mat<R> MatOps<R>::transpose(Mat<R> matrix) {
    assert(matrix.dims().size() > 1);
    Mat<R> out (
        matrix.dims(1),
        matrix.dims(0),
        false);
    out.w() = matrix.w().transpose();
    if (graph::backprop_enabled)
        graph::emplace_back([matrix, out](){
            GRAD(matrix).noalias() += (out.dw()).transpose();
        });
    return out;
}

template<typename R>
Mat<R> MatOps<R>::tanh(Mat<R> matrix) {
    auto out = Mat<R>::empty_like(matrix);
    out.w() = matrix.w().unaryExpr(utils::tanh_operator<R>());
    if (graph::backprop_enabled)
        graph::emplace_back([matrix, out](){
            GRAD(matrix).noalias() += (out.w().unaryExpr(utils::dtanh_operator<R>()).array() * out.dw().array()).matrix();
        });
    return out;
}

template<typename R>
Mat<R> MatOps<R>::relu(Mat<R> matrix) {
    auto out = Mat<R>::empty_like(matrix);
    out.w() = matrix.w().unaryExpr(utils::relu_operator<R>());
    if (graph::backprop_enabled)
        graph::emplace_back([matrix, out](){
            GRAD(matrix).noalias() += (out.w().unaryExpr(utils::max_operator<R>()).array() * out.dw().array()).matrix();
        });
    return out;
}

template<typename R>
Mat<R> MatOps<R>::abs(Mat<R> matrix) {
    auto out = Mat<R>::empty_like(matrix);
    out.w() = matrix.w().array().abs().matrix();
    if (graph::backprop_enabled)
        graph::emplace_back([matrix, out](){
            GRAD(matrix).noalias() += (matrix.w().unaryExpr(utils::sign_operator<R>()).array() * out.dw().array()).matrix();
        });
    return out;
}

template<typename R>
Mat<R> MatOps<R>::mul(
    Mat<R> matrix1,
    Mat<R> matrix2) {
    assert2(matrix1.dims(1) == matrix2.dims(0), "matmul dimensions misaligned.");
    Mat<R> out (
        matrix1.dims(0),
        matrix2.dims(1),
        false);
    out.w() = matrix1.w() * matrix2.w();
    if (graph::backprop_enabled)
        graph::emplace_back([matrix1, matrix2, out](){
            GRAD(matrix1).noalias() += (out.dw()) * ((matrix2.w()).transpose());
            GRAD(matrix2).noalias() += matrix1.w().transpose() * (out.dw());
        });
    return out;
}

template<typename R>
Mat<R> MatOps<R>::quadratic_form(
    Mat<R> left,
    Mat<R> weights,
    Mat<R> right) {

    assert2(weights.dims(1) == right.dims(0), "Quadratic form right matrix has wrong dimensions.");
    assert2(left.dims(0) == weights.dims(0) , "Quadratic form left matrix has wrong dimensions.");

    Mat<R> out (
        left.dims(1),
        right.dims(1),
        false);
    if (graph::backprop_enabled) {
        auto left_side_mul = std::make_shared<Eigen::Matrix<R, Eigen::Dynamic, Eigen::Dynamic>>(
            left.dims(1), weights.dims(1)
        );
        (*left_side_mul) = left.w().transpose() * weights.w();
        out.w() = (*left_side_mul) * right.w();
        graph::emplace_back([left_side_mul, left, weights, right, out](){
            GRAD(right).noalias() += (*left_side_mul).transpose() * out.dw();
            auto LeftT_dot_weights_grad = out.dw() * (right.w().transpose());
            GRAD(left).noalias() += (
                LeftT_dot_weights_grad * weights.w().transpose()
            ).transpose();
            GRAD(weights).noalias() += left.w() * LeftT_dot_weights_grad;
        });
    } else {
        out.w() = left.w().transpose() * weights.w() * right.w();
    }
    return out;
}

template<typename R>
Mat<R> MatOps<R>::mul_with_bias(
    Mat<R> matrix1,
    Mat<R> matrix2,
    Mat<R> bias) {
    assert2(matrix1.dims(1) == matrix2.dims(0), "matmul dimensions misaligned.");
    if (matrix1.dims(0) != bias.dims(0) || bias.dims(1) != 1)
        throw std::invalid_argument("Matrices cannot be multiplied with broadcast, they do not have the same dimensions.");
    Mat<R> out (
            matrix1.dims(0),
            matrix2.dims(1),
            false);
    out.w() = ((matrix1.w() * matrix2.w()).colwise() + bias.w().col(0)).matrix();
    if (graph::backprop_enabled)
        graph::emplace_back([matrix1, matrix2, bias, out]() {
            GRAD(matrix1).noalias() += (out.dw()) * ((matrix2.w()).transpose());
            GRAD(matrix2).noalias() += matrix1.w().transpose() * (out.dw());
            GRAD(bias).noalias()    += out.dw().rowwise().sum().matrix();
        });
    return out;
}

template<typename R>
Mat<R> MatOps<R>::mul_add_broadcast_mul_with_bias(
    Mat<R> matrix1,
    Mat<R> input_to_1,
    Mat<R> matrix2,
    Mat<R> input_to_2,
    Mat<R> bias) {
    assert2(matrix1.dims(1) == input_to_1.dims(0), "matmul 1 dimensions misaligned.");
    if (matrix2.dims(1) != input_to_2.dims(0))
        throw std::invalid_argument("matmul 2 dimensions misaligned.");
    if (matrix2.dims(0) != bias.dims(0) || matrix1.dims(0) != bias.dims(0) || input_to_1.dims(1) != 1 || bias.dims(1) != 1)
        throw std::invalid_argument("Matrices cannot be shigamizood, they do not have the same dimensions.");
    Mat<R> out (
            matrix1.dims(0),
            input_to_2.dims(1),
            false);
    // both input to 1 and bias are columns,
    // so we add both of those before adding the true matrix
    // product in broadcasted form
    out.w() = (
          (
              (
                  (matrix2.w() * input_to_2.w())
              )
          ).colwise() + (bias.w() + (matrix1.w() * input_to_1.w())).col(0)
      ).matrix();
    if (graph::backprop_enabled)
        graph::emplace_back([matrix1, input_to_1, matrix2, input_to_2, bias, out] () {
            // first multiply:
            // broadcasting input means taking outer product here:
            GRAD(matrix1) += ((out.dw()).rowwise().sum() * ((input_to_1.w()).transpose()));
            // broadcasting output means sum after the reverse product here:
            GRAD(input_to_1).noalias() += (
                matrix1.w().transpose() * (out.dw())
            ).rowwise().sum();
            // second multiply:
            GRAD(matrix2).noalias() += (out.dw()) * ((input_to_2.w()).transpose());

            GRAD(input_to_2).noalias() += matrix2.w().transpose() * (out.dw());
            // bias vector:
            GRAD(bias).noalias() += out.dw().rowwise().sum();
        });
    return out;
}


template<typename R>
Mat<R> MatOps<R>::mul_add_mul_with_bias(std::initializer_list<Mat<R>> matrices) {
    vector<Mat<R>> matrices_vector(matrices);
    return mul_add_mul_with_bias(matrices_vector);
}

template<typename R>
Mat<R> MatOps<R>::mul_add_mul_with_bias(const vector<Mat<R>>& matrices) {
    // broacast to largest input size
    dim_t max_broadcast = matrices[1].dims(1);
    for (auto matrices_ptr = matrices.begin()+1; matrices_ptr < matrices.end(); matrices_ptr+=2) {
        max_broadcast = std::max(max_broadcast, matrices_ptr->dims(1));
    }

    Mat<R> out(
            matrices[0].dims(0),
            max_broadcast,
            true);
    DEBUG_ASSERT_MAT_NOT_NAN(out)
    auto matrices_ptr = matrices.begin();
    while (matrices_ptr != (matrices.end() - 1)) {
        DEBUG_ASSERT_MAT_NOT_NAN(*matrices_ptr)
        DEBUG_ASSERT_MAT_NOT_NAN(*(matrices_ptr + 1))
        DEBUG_ASSERT_MAT_NOT_NAN(out)
        // inputs must either match the broadcasted size, or be broadcastable by having their
        // inner dimension be 1 (a column vector essentially)
        assert(((matrices_ptr+1)->dims(1) == max_broadcast) || ((matrices_ptr+1)->dims(1) == 1));
        if ((matrices_ptr+1)->dims(1) == max_broadcast) {
            out.w() += (*matrices_ptr).w() * (*(matrices_ptr + 1)).w();
        } else {
            auto el = ((*matrices_ptr).w() * (*(matrices_ptr + 1)).w());
            out.w().colwise() += el.col(0);
        }
        DEBUG_ASSERT_MAT_NOT_NAN(out)
        matrices_ptr+=2;
    }

    DEBUG_ASSERT_NOT_NAN(matrices.back().w());
    out.w().colwise() += matrices.back().w().col(0);
    if (graph::backprop_enabled)
        graph::emplace_back([matrices, out, max_broadcast](){
            auto matrices_ptr = matrices.begin();
            while (matrices_ptr != (matrices.end() - 1)) {
                if ((matrices_ptr+1)->dims(1) == max_broadcast) {
                    GRAD(*matrices_ptr).noalias()   += out.dw() * (*(matrices_ptr+1)).w().transpose();
                    GRAD(*(matrices_ptr+1)).noalias() += (*matrices_ptr).w().transpose() * (out.dw());
                } else {
                    // broadcasting input means taking outer product here:
                    GRAD(*matrices_ptr).noalias() += (out.dw().rowwise().sum() * ((*(matrices_ptr+1)).w().transpose()));
                    // broadcasting output means sum after the reverse product here:
                    GRAD(*(matrices_ptr+1)).noalias() += (
                        (*matrices_ptr).w().transpose() * out.dw()
                    ).rowwise().sum();
                }
                matrices_ptr+=2;
            }
            GRAD(matrices.back()).noalias() += out.dw().rowwise().sum();
        });

    DEBUG_ASSERT_NOT_NAN(out.w());
    return out;
}

// operation of the form (A * x + B * y) + C, called with mul_add_mul_with_bias(A, x, B, y, C)
template<typename R>
Mat<R> MatOps<R>::mul_add_mul_with_bias(
    Mat<R> matrix1,
    Mat<R> input_to_1,
    Mat<R> matrix2,
    Mat<R> input_to_2,
    Mat<R> bias) {
    DEBUG_ASSERT_NOT_NAN(bias.w());
    assert2(matrix1.dims(1) == input_to_1.dims(0), "matmul 1 dimensions misaligned.");
    if (matrix2.dims(1) != input_to_2.dims(0))
        throw std::invalid_argument("matmul 2 dimensions misaligned.");
    if (matrix2.dims(0) != bias.dims(0) || matrix1.dims(0) != bias.dims(0) || bias.dims(1) != 1)
        throw std::invalid_argument("Matrices cannot be shigamizood, they do not have the same dimensions.");
    if (input_to_1.dims(1) != input_to_2.dims(1)) {
        if (input_to_1.dims(1) == 1) {
            return mul_add_broadcast_mul_with_bias(matrix1, input_to_1, matrix2, input_to_2, bias);
        }
        return mul_add_broadcast_mul_with_bias(matrix2, input_to_2, matrix1, input_to_1, bias);
    }
    Mat<R> out (
            matrix1.dims(0),
            input_to_1.dims(1),
            false);
    out.w() = (
                  (
                      (
                          (matrix1.w() * input_to_1.w()) +
                          (matrix2.w() * input_to_2.w())
                      )
                  ).colwise() + bias.w().col(0)
              ).matrix();
    if (graph::backprop_enabled)
        graph::emplace_back([matrix1, input_to_1, matrix2, input_to_2, bias, out](){
            // first multiply:
            // broadcasting input means taking outer product here:
            GRAD(matrix1)               += (out.dw() * (input_to_1.w()).transpose());
            // broadcasting output means sum after the reverse product here:
            GRAD(input_to_1).noalias() += matrix1.w().transpose() * (out.dw());
            // second multiply:
            GRAD(matrix2).noalias()     += (out.dw()) * (input_to_2.w()).transpose();

            GRAD(input_to_2).noalias()  += matrix2.w().transpose() * (out.dw());
            // bias vector:
            GRAD(bias).noalias()        += out.dw().rowwise().sum();
        });
    return out;
}

template<typename R>
Mat<R> MatOps<R>::rows_pluck(
        Mat<R> matrix,
        Indexing::Index indices
        ) {
    Mat<R> out (
        matrix.dims(1),
        indices.size(),
        false);

    for (std::size_t offset = 0; offset < indices.size(); ++offset) {
        out.w().col(offset) = matrix.w().row(indices[offset]).transpose();
    }
    if (graph::backprop_enabled) {
        graph::emplace_back([matrix, out, indices](){
            auto index_ptr = indices.data();
            for (std::size_t i = 0; i < out.dims(1); ++i) {
                // for each row do the same operation as for row_pluck:
                GRAD(matrix).row(*index_ptr).noalias() += out.dw().col(i).transpose();
                index_ptr++;
            }
        });
    }
    return out;
}

template<typename R>
Mat<R> MatOps<R>::dropout(
    Mat<R> matrix,
    R drop_prob) {

    assert(0.0 <= drop_prob && drop_prob <= 1.0);

    // no dropout happens.
    if (drop_prob < 1e-6)
        return matrix;

    auto out = Mat<R>::empty_like(matrix);

    auto bool_mat = std::make_shared<Eigen::Matrix<R, Eigen::Dynamic, Eigen::Dynamic>>(matrix.dims(0), matrix.dims(1));

    std::default_random_engine generator;
    std::bernoulli_distribution distribution(1.0 - drop_prob);
    std::random_device rd;
    generator.seed(rd());

    auto data_ptr = matrix.w().data();
    auto out_ptr  = out.w().data();
    auto bool_ptr = bool_mat->data();

    for (int i = 0; i < matrix.number_of_elements();++i) {
        (*bool_ptr) = distribution(generator) ? 1.0 : 0.0;
        (*out_ptr) = (*bool_ptr) > 0 ? *data_ptr : 0.0;
        out_ptr++;
        data_ptr++;
        bool_ptr++;
    }

    if (graph::backprop_enabled) {
        graph::emplace_back([matrix, out, bool_mat](){
            GRAD(matrix) += (out.dw().array() * (*bool_mat).array()).matrix();
        });
    }
    return out;
}

template<typename R>
Mat<R> MatOps<R>::dropout_normalized(
    Mat<R> matrix,
    R drop_prob) {

    assert(0.0 <= drop_prob && drop_prob <= 1.0);

    // no dropout happens.
    if (drop_prob < 1e-6)
        return matrix;

    auto out = Mat<R>::empty_like(matrix);

    auto bool_mat = std::make_shared<Eigen::Matrix<R, Eigen::Dynamic, Eigen::Dynamic>>(matrix.dims(0), matrix.dims(1));

    std::default_random_engine generator;
    std::bernoulli_distribution distribution(1.0 - drop_prob);
    std::random_device rd;
    generator.seed(rd());

    auto data_ptr = matrix.w().data();
    auto out_ptr  = out.w().data();
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
        graph::emplace_back([matrix, out, bool_mat](){
            GRAD(matrix) += (out.dw().array() * (*bool_mat).array()).matrix();
        });
    }
    return out;
}

template<typename R>
vector<Mat<R>> MatOps<R>::dropout_normalized(
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
vector<Mat<R>> MatOps<R>::dropout(
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
Mat<R> MatOps<R>::fast_dropout(Mat<R> matrix) {
    auto out = Mat<R>::empty_like(matrix);

    auto randn_mat = std::make_shared<Eigen::Matrix<R, Eigen::Dynamic, Eigen::Dynamic>>(matrix.dims(0), matrix.dims(1));

    std::default_random_engine generator;
    std::normal_distribution<R> distribution(1.0, 1.0);
    std::random_device rd;
    generator.seed(rd());

    auto data_ptr = matrix.w().data();
    auto out_ptr  = out.w().data();
    auto randn_ptr = randn_mat->data();

    for (unsigned int i = 0; i < matrix.number_of_elements();++i) {
        (*randn_ptr) = distribution(generator);
        (*out_ptr) = (*randn_ptr) * *data_ptr;
        out_ptr++;
        data_ptr++;
        randn_ptr++;
    }

    if (graph::backprop_enabled) {
        graph::emplace_back([matrix, out, randn_mat](){
            GRAD(matrix) += (out.dw().array() * (*randn_mat).array()).matrix();
        });
    }
    return out;
}

template<typename R>
Mat<R> MatOps<R>::rows_cols_pluck(
        Mat<R> matrix,
        Indexing::Index row_indices,
        Indexing::Index col_indices) {
    if (row_indices.size() != col_indices.size())
        throw std::invalid_argument("Cannot pluck column row pairs, not the same amount of row and column indices.");
        Mat<R> out (
            1,
            row_indices.size(),
            false);
        for (int offset = 0; offset < row_indices.size(); ++offset)
            out.w()(offset) = matrix.w()(row_indices[offset], col_indices[offset]);
    if (graph::backprop_enabled && !matrix.constant) {
        graph::emplace_back([matrix, out, row_indices, col_indices](){
            auto row_index_ptr = row_indices.data();
            auto col_index_ptr = col_indices.data();
            for (int i = 0; i < out.dims(1); ++i) {
                // for each row do the same operation as for row_pluck:
                matrix.dw()(*row_index_ptr, *col_index_ptr) += out.dw()(i);
                row_index_ptr++;
                col_index_ptr++;
            }
        });
    }
    return out;
}

template<typename R>
Mat<R> MatOps<R>::row_pluck(
        Mat<R> matrix,
        int row) {
    Mat<R> out (matrix.dims(1), 1, false);
    out.w() = matrix.w().row(row).transpose();
    if (graph::backprop_enabled)
        graph::emplace_back([matrix, out, row]() {
            GRAD(matrix).row(row).noalias() += out.dw().col(0).transpose();
        });
    return out;
}

template<typename R>
Mat<R> MatOps<R>::col_pluck(
        Mat<R> matrix,
        int col) {
    Mat<R> out (matrix.dims(0), 1, false);
    out.w() = matrix.w().col(col);
    if (graph::backprop_enabled)
        graph::emplace_back([matrix, out, col]() {
            GRAD(matrix).col(col).noalias() += out.dw().col(0).transpose();
        });
    return out;
}

template<typename R>
Mat<R> MatOps<R>::consider_constant_if(
        Mat<R> matrix,
        bool should_consider_constant) {
    if (should_consider_constant)
        return consider_constant(matrix);
    return matrix;
}

template<typename R>
Mat<R> MatOps<R>::consider_constant(
        Mat<R> matrix
        ) {
    // perform a copy of the matrix that references
    // everything and owns nothing. A true nomad.
    Mat<R> out(matrix, false, false);
    out.constant = true;
    return out;
}

template class MatOps<float>;
template class MatOps<double>;
