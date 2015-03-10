#ifndef RECURRENT_MAT_H
#define RECURRENT_MAT_H

#include <Eigen/Eigen>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <unordered_map>

#include "core/cnpy.h"
#include "core/Index.h"
#include "core/Tape.h"
#include "core/utils.h"


DECLARE_bool(eigen_parallel);

#define MAT Mat<R>
#define SHARED_MAT std::shared_ptr<MAT>

#define EPS 1e-9

typedef std::size_t random_t;

/**
Mat
---

Main matrix class for this library. The Mat
class contains two pieces of memory, `w`
and `dw`. The first is the actual weights or
values associated with this matrix, and the
second is the local contribution to the
objective function (or ∂E/∂Mat). This local
contribution can then be used in
backpropagation.

Mat is used almost everywhere in the library
except in `utils`.

Note: ideally this class would generalize to
higher and lower dimensions. For instance
see `Graph::sum` and `Graph::mean`, or
`Mat::grad` methods to see where a Scalar
type would be useful (today this is a 1x1
matrix -- mathematically sound, but this
is inefficient, and inadequate for good
broadcasting across other operations).

**/


template<typename R>
class Mat : public std::enable_shared_from_this<Mat<R>> {
    private:
        static std::atomic<int> next_matrix;
        typedef Eigen::Matrix<R, Eigen::Dynamic, Eigen::Dynamic> eigen_mat;
        typedef Eigen::Map<eigen_mat> eigen_mat_view;
        eigen_mat _w;
        eigen_mat _dw;
    public:
        typedef unsigned int dim_t;

        std::vector<dim_t> dims;
        mutable eigen_mat_view w;
        bool sparse;
        std::shared_ptr<std::vector<uint>> sparse_row_keys;
        mutable eigen_mat_view dw;
        std::shared_ptr<std::string> name = nullptr;
        const random_t random_id;
        Mat (dim_t n, dim_t d);
        Mat (dim_t n, dim_t d, bool empty);
        /**
        Mat<R>::Mat<R>
        --------------

        A copy constructor that perform shallow and deep
        copies of a Mat.

        Key usage is for Hogwild style training of parameters
        where different computation threads share memory for
        the parameters but each compute their own gradients.
        The gradients are kept in separate `dw` memory buffers
        but `w` buffers are shared amongst threads.

        For usage see `Solver::Adadelta`,
                      `examples/character_prediction.cpp`

        Inputs
        ------

        const Mat<R>& m : matrix to copy or point to
            bool copy_w : whether matrix parameters should be copied over, or shared
                          between matrices
           bool copy_dw : whether matrix gradient parameters should be copied over,
                          or shared (Note: it is unclear when `dw` should be shared,
                          proceed with caution).

        Outputs
        -------

        Mat<R> out : deep or shallow copy of m
        **/
        Mat (const Mat<R>& m, bool copy_w, bool copy_dw);
        void print() const;

        /**
        Adds 1 to the gradient (`dw`) of every element in this Matrix as long
        as the matrix is 1x1 (a scalar).

        Useful for computing cross entropy, mean squared error, and other
        loss functions in vanilla ML fashion.

        **/
        void grad();
        unsigned int number_of_elements() const;

        ~Mat();
        /*
        Set Name
        --------

        Used for giving names to matrices for debugging or convenience purposes,
        but the names have no bearing on computation or identification in
        lookup tables;

        Inputs
        ------

        std::string& name : name the Mat should take on

        */
        void set_name(std::string& newname);
        /*
        Set Name
        --------
        See `Mat<R>::set_name` above
        */
        void set_name(char* newname);
        /*
        Set Name
        --------
        See `Mat<R>::set_name` above
        */
        void set_name(const char* newname);
        // random matrices:
        /*
        Mat<R>::Mat<R>
        --------------

        Matrix constructor using a zero mean
        normal distribution with a user provided
        standard deviation.

        Inputs
        ------

        int _n : number of rows
        int _d : number of columns
         R std : standard deviation for normal distribution

        Outputs
        -------

        Mat<R> out : the matrix filled with random numbers from ~ N(0, std^2)

        See `Mat<R>::Mat(int, int, R, R)` for uniform distribution (below).
        */
        Mat (dim_t n, dim_t d, R std);
        /*
        Mat<R>::Mat<R>
        --------------

        Matrix constructor using a uniform
        distribution with user defined min
        and max support.

        Inputs
        ------

          int _n : number of rows
          int _d : number of columns
         R lower : minimum of uniform distribution
         R upper : maximum of uniform distribution

        Outputs
        -------

        Mat<R> out : the matrix filled with random numbers from ~U(lower, upper)

        See `Mat<R>::Mat(int, int, R)` for normal distribution (above)
        */
        Mat (dim_t n, dim_t d, R lower, R upper);

        void npy_save(std::string fname, std::string mode = "w");
        void npy_save(FILE*);
        void npy_load(std::string fname);
        void npy_load(FILE*);
        void npy_load(cnpy::NpyArray&);
        Mat (std::string fname);
        static Mat RandMat(dim_t n, dim_t d, R std);
        static Mat Empty(dim_t n, dim_t d);
        /**
        Shallow Copy
        ------------

        A copy constructor that perform shallow copies of a Mat.

        Key usage is for Hogwild style training of parameters
        where different computation threads share memory for
        the parameters but each compute their own gradients.
        The gradients are kept in separate `dw` memory buffers
        but `w` buffers are shared amongst threads.

        For usage see `Mat<R>::Mat<R>(const Mat&, bool, bool)`, `examples/character_prediction.cpp`

        Inputs
        ------

        const Mat<R>& m : matrix that will own the underlying memory
                          for `w`

        Outputs
        -------

        Mat<R> out : shallow copy of m
        **/
        static Mat shallow_copy(const Mat&);
        operator std::string() const;







        SHARED_MAT eltmul_broadcast(SHARED_MAT);
        SHARED_MAT eltmul(SHARED_MAT);
        SHARED_MAT eltmul(R);

        /**
        Element Multiplication Broadcast Rowwise
        ----------------------------------------

        To treat the special case of a row vector that must be multiplied
        with a matrix, rowwise, the we ensure that the row_vector has only
        one row, and the number of columns of this row vector is equal to
        the number of rows of matrix1.

        Inputs
        ------

        SHARED_MAT matrix1    : the matrix to multiply row wise
        SHARED_MAT row_vector : the row vector to multiply with each row
                                of matrix1 individually.

        Outputs
        -------

        SHARED_MAT out : the rowwise multiply of matrix1 with row_vector.
        **/
        SHARED_MAT eltmul_broadcast_rowwise(SHARED_MAT);
        /**
        Element Multiplication Rowwise
        ------------------------------

        The more general case is the element wise multiplication of two
        matrices A and B, with B transposed:

        > out = A * B^T

        Inputs
        ------

        SHARED_MAT matrix1    : the matrix to multiply
        SHARED_MAT matrix2    : the matrix to multiply after transposing

        Outputs
        -------

        SHARED_MAT out : the element wise product of matrix1 and matrix2^T

        **/
        SHARED_MAT eltmul_rowwise(SHARED_MAT);
        static SHARED_MAT mul_with_bias(SHARED_MAT, SHARED_MAT, SHARED_MAT);
        // operation of the form (A * x + B * y) + C, called with mul_add_mul_with_bias(A, x, B, y, C)
        static SHARED_MAT mul_add_mul_with_bias(SHARED_MAT, SHARED_MAT, SHARED_MAT, SHARED_MAT, SHARED_MAT);
        static SHARED_MAT mul_add_mul_with_bias(std::initializer_list<SHARED_MAT>);
        static SHARED_MAT mul_add_mul_with_bias(const std::vector<SHARED_MAT>&);
        // operation of the form (A * x + B * y) + C, called with mul_add_mul_with_bias(A, x, B, y, C)
        // and with caveat that x is actually a column, and should be broadcasted
        static SHARED_MAT mul_add_broadcast_mul_with_bias(SHARED_MAT, SHARED_MAT, SHARED_MAT, SHARED_MAT, SHARED_MAT);
        SHARED_MAT add_broadcast(SHARED_MAT);

        SHARED_MAT add(SHARED_MAT);
        SHARED_MAT sub(SHARED_MAT);
        SHARED_MAT sub_broadcast(SHARED_MAT);
        SHARED_MAT sub_broadcast_reversed(SHARED_MAT);

        static SHARED_MAT add(std::initializer_list<SHARED_MAT>);
        SHARED_MAT square();

        SHARED_MAT sum();

        SHARED_MAT mean();
        SHARED_MAT log();
        SHARED_MAT exp();

        static SHARED_MAT binary_cross_entropy(SHARED_MAT, R);

        static SHARED_MAT sigmoid_binary_cross_entropy(SHARED_MAT, R);
        static SHARED_MAT cross_entropy(SHARED_MAT, uint answer_idx);
        static SHARED_MAT softmax_cross_entropy(SHARED_MAT matrix, uint answer_idx);
        static SHARED_MAT hstack(SHARED_MAT, SHARED_MAT);
        static SHARED_MAT hstack(std::initializer_list<SHARED_MAT>);
        static SHARED_MAT hstack(const std::vector<SHARED_MAT>&);
        static SHARED_MAT vstack(SHARED_MAT, SHARED_MAT);
        static SHARED_MAT vstack(std::initializer_list<SHARED_MAT>);
        static SHARED_MAT vstack(const std::vector<SHARED_MAT>&);

        SHARED_MAT sigmoid();
        static SHARED_MAT softmax(SHARED_MAT, R temperature=1.0);
        SHARED_MAT steep_sigmoid(R aggressiveness = 3.75);
        SHARED_MAT T();
        SHARED_MAT tanh();
        SHARED_MAT relu();
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

        SHARED_MAT matrix : where to apply the noise
              R drop_prob : likelihood that an element of the matrix
                            goes to 0

        Outputs
        -------

        SHARED_MAT out : noisy matrix

        **/
        static SHARED_MAT dropout(SHARED_MAT, R);
        static SHARED_MAT dropout_normalized(SHARED_MAT, R);

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

        SHARED_MAT matrix : where to apply the noise

        Outputs
        -------

        SHARED_MAT out : noisy matrix

        **/
        static SHARED_MAT fast_dropout(SHARED_MAT);
        SHARED_MAT mul(SHARED_MAT);
        SHARED_MAT rows_pluck(Indexing::Index);
        SHARED_MAT rows_cols_pluck(Indexing::Index, Indexing::Index);
        SHARED_MAT row_pluck(int);











};

template<typename R>
std::ostream& operator<<(std::ostream&, const Mat<R>&);

// define hash code for matrices:
namespace std {
        template <typename R> struct hash<Mat<R>> {
                std::size_t operator()(const Mat<R>&) const;
        };
}

namespace utils {
    template<typename R>
    void save_matrices(std::vector<std::shared_ptr<Mat<R>>>&, std::string);

    template<typename R>
    void load_matrices(std::vector<std::shared_ptr<Mat<R>>>&, std::string);
}

template <typename R>
bool operator!=(const Mat<R>&, const Mat<R>&);

template <typename R>
bool operator==(const Mat<R>&, const Mat<R>&);

template<typename R>
int argmax(std::shared_ptr<Mat<R>>);

template<typename R>
int argmax_slice(std::shared_ptr<Mat<R>>, int, int);

#endif
