#ifndef CORE_MAT_OPS_H
#define CORE_MAT_OPS_H

#include <initializer_list>
#include <vector>
#include <memory>

#include <Eigen/Eigen>
#include "dali/mat/Mat.h"
#include "dali/utils.h"

template<typename R> class Mat;
namespace Indexing {
    class Index;
}

// this is only a class so that it is easier to instantiate
// templates. In principle this should be a namespace.
template<typename R>
struct MatOps {
    static Mat<R> eltmul_broadcast(Mat<R>, Mat<R>);
    static Mat<R> eltdivide_broadcast(Mat<R>, Mat<R>);
    static Mat<R> eltdivide_broadcast_reversed(Mat<R>, Mat<R>);
    static Mat<R> eltmul(Mat<R>, Mat<R>);
    static Mat<R> eltmul(Mat<R>, R);
    static Mat<R> eltdivide(Mat<R>, Mat<R>);
    static Mat<R> eltdivide(Mat<R>, R);

    /**
    Element Multiplication Broadcast Rowwise
    ----------------------------------------

    To treat the special case of a row vector that must be multiplied
    with a matrix, rowwise, the we ensure that the row_vector has only
    one row, and the number of columns of this row vector is equal to
    the number of rows of matrix1.

    Inputs
    ------

    Mat<R> matrix1    : the matrix to multiply row wise
    Mat<R> row_vector : the row vector to multiply with each row
                            of matrix1 individually.

    Outputs
    -------

    Mat<R> out : the rowwise multiply of matrix1 with row_vector.
    **/
    static Mat<R> eltmul_broadcast_rowwise(Mat<R>, Mat<R>);
    /**
    Element Multiplication Rowwise
    ------------------------------

    The more general case is the element wise multiplication of two
    matrices A and B, with B transposed:

    > out = A * B^T

    Inputs
    ------

    Mat<R> matrix1    : the matrix to multiply
    Mat<R> matrix2    : the matrix to multiply after transposing

    Outputs
    -------

    Mat<R> out : the element wise product of matrix1 and matrix2^T

    **/
    static Mat<R> eltmul_rowwise(Mat<R>, Mat<R>);
    static Mat<R> mul_with_bias(Mat<R>, Mat<R>, Mat<R>);
    // operation of the form (A * x + B * y) + C, called with mul_add_mul_with_bias(A, x, B, y, C)
    static Mat<R> mul_add_mul_with_bias(Mat<R>, Mat<R>, Mat<R>, Mat<R>, Mat<R>);
    static Mat<R> mul_add_mul_with_bias(std::initializer_list<Mat<R>>);
    static Mat<R> mul_add_mul_with_bias(const std::vector<Mat<R>>&);
    // operation of the form (A * x + B * y) + C, called with mul_add_mul_with_bias(A, x, B, y, C)
    // and with caveat that x is actually a column, and should be broadcasted
    static Mat<R> mul_add_broadcast_mul_with_bias(Mat<R>, Mat<R>, Mat<R>, Mat<R>, Mat<R>);
    static Mat<R> add_broadcast(Mat<R>, Mat<R>);

    static Mat<R> add(Mat<R>, Mat<R>);
    static Mat<R> sub(Mat<R>, Mat<R>);
    static Mat<R> add(Mat<R>, R);
    static Mat<R> sub_broadcast(Mat<R>, Mat<R>);
    static Mat<R> sub_broadcast_reversed(Mat<R>, Mat<R>);
    static Mat<R> sub_broadcast_reversed(Mat<R>, R);

    static Mat<R> add(std::initializer_list<Mat<R>>);
    static Mat<R> add(const std::vector<Mat<R>>&);
    static Mat<R> square(Mat<R>);

    static Mat<R> sum(Mat<R>);

    static Mat<R> mean(Mat<R>);
    static Mat<R> log(Mat<R>);
    static Mat<R> exp(Mat<R>);

    /**
    Binary Cross entropy error:

    D_{KL}(p || q) = sum_i { (1-p_i) * log(1 - q_i) -p_i log (q_i) }

    **/
    static Mat<R> binary_cross_entropy(Mat<R>, R);
    static Mat<R> sigmoid_binary_cross_entropy(Mat<R>, R);
    static Mat<R> cross_entropy(Mat<R>, uint answer_idx);
    static Mat<R> softmax_cross_entropy(Mat<R> matrix, uint answer_idx);
    static Mat<R> hstack(Mat<R>, Mat<R>);
    static Mat<R> hstack(std::initializer_list<Mat<R>>);
    static Mat<R> hstack(const std::vector<Mat<R>>&);
    static Mat<R> vstack(Mat<R>, Mat<R>);
    static Mat<R> vstack(std::initializer_list<Mat<R>>);
    static Mat<R> vstack(const std::vector<Mat<R>>&);

    static Mat<R> sigmoid(Mat<R>);
    static Mat<R> softmax(Mat<R>, R temperature=1.0);
    static Mat<R> softmax_no_grad(Mat<R>, R temperature = 1.0);
    static Mat<R> steep_sigmoid(Mat<R>, R aggressiveness = 3.75);
    static Mat<R> transpose(Mat<R>);
    static Mat<R> tanh(Mat<R>);
    static Mat<R> relu(Mat<R>);
    /**
    Dropout
    -------

    Apply bernoulli noise to a matrix (e.g. to regularize a
    neural network).

    Deep neural nets with a large number of parameters are
    very powerful machine learning systems. However, overfitting
    is a serious problem in such networks. Large networks are
    also slow to use, making it difficult to deal with overfitting
    by combining the predictions of many different large neural
    nets at test time. Dropout is a technique for addressing
    this problem. The key idea is to randomly drop units (along
    with their connections) from the neural network during training.
    This prevents units from co-adapting too much. During training,
    dropout samples from an exponential number of different
    “thinned” networks. At test time, it is easy to approximate
    the effect of averaging the predictions of all these thinned
    networks by simply using a single unthinned network that has
    smaller weights. This significantly reduces overfitting and
    gives major improvements over other regularization methods.

    - Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky,
    Ilya Sutskever, Ruslan Salakhutdinov, "Dropout: A Simple Way
    to Prevent Neural Networks from Overfitting," JMLR 2014

    Inputs
    ------

    Mat<R> matrix : where to apply the noise
          R drop_prob : likelihood that an element of the matrix
                        goes to 0

    Outputs
    -------

    Mat<R> out : noisy matrix

    **/
    static Mat<R> dropout(Mat<R>, R);
    static Mat<R> dropout_normalized(Mat<R>, R);

    /**
    Fast Dropout
    ------------

    Apply Gaussian Noise a standard deviation of 1 and a
    mean of 1 to a matrix (e.g. to regularize it)

    Preventing feature co-adaptation by encouraging independent
    contributions from differ- ent features often improves
    classification and regression performance. Dropout training
    (Hinton et al., 2012) does this by randomly dropping out
    (zeroing) hidden units and input features during training
    of neural networks. However, repeatedly sampling a random
    subset of input features makes training much slower. Based
    on an examination of the implied objective function of dropout
    training, we show how to do fast dropout training by sampling
    from or integrating a Gaussian approximation, instead of
    doing Monte Carlo optimization of this objective. This
    approximation, justified by the central limit theorem and
    empirical evidence, gives an order of magnitude speedup and
    more stability.
    - Sida I. Wang, Christopher D. Manning, "Fast dropout training",
    ICML 2013

    Also see this Github gist:
    https://gist.github.com/SnippyHolloW/8a0f820261926e2f41cc

    Inputs
    ------

    Mat<R> matrix : where to apply the noise

    Outputs
    -------

    Mat<R> out : noisy matrix

    **/
    static Mat<R> fast_dropout(Mat<R>);
    static Mat<R> mul(Mat<R>, Mat<R>);
    static Mat<R> rows_pluck(Mat<R>, Indexing::Index);
    static Mat<R> rows_cols_pluck(Mat<R>, Indexing::Index, Indexing::Index);
    static Mat<R> row_pluck(Mat<R>, int);
    static Mat<R> col_pluck(Mat<R>, int);
    static Mat<R> pow(Mat<R>, R);
    static Mat<R> fill(Mat<R>, R);
    static Mat<R> sqrt(Mat<R>);
    static Mat<R> elt_inv(Mat<R>);
    static Mat<R> conv2d(Mat<R> image, Mat<R> kernel);
    static Mat<R> conv1d(Mat<R> image, Mat<R> kernel);
    static Mat<R> conv1d(Mat<R> image, Mat<R> kernel, bool pad);
    static Mat<R> conv1d(Mat<R> image, const std::vector<Mat<R>>& kernels);
    static Mat<R> conv1d(Mat<R> image, const std::vector<Mat<R>>& kernels, bool pad);

    static Mat<R> consider_constant(Mat<R>);
    static Mat<R> consider_constant_if(Mat<R>, bool should_consider_constant);
};


#endif
