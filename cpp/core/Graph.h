#ifndef GRAPH_MAT_H
#define GRAPH_MAT_H

#include "Mat.h"
//#include "Backward.h"
#include <memory>
#include <sstream>
#include <string>
#include <functional>

template<typename T> class Graph {

        std::vector<std::function<void()>>  backprop;
        typedef Mat<T>                      mat;
        typedef std::shared_ptr<mat> shared_mat;
        public:
                bool                 needs_backprop;
                Graph (bool);
                Graph ();
                int backprop_size() const;
                void backward ();
                shared_mat eltmul_broadcast(shared_mat, shared_mat);
                shared_mat eltmul(shared_mat, shared_mat);
                shared_mat eltmul(shared_mat, T);

                /**
                Element Multiplication Broadcast Rowwise
                ----------------------------------------

                To treat the special case of a row vector that must be multiplied
                with a matrix, rowwise, the we ensure that the row_vector has only
                one row, and the number of columns of this row vector is equal to
                the number of rows of matrix1.

                Inputs
                ------

                shared_mat matrix1    : the matrix to multiply row wise
                shared_mat row_vector : the row vector to multiply with each row
                                        of matrix1 individually.

                Outputs
                -------

                shared_mat out : the rowwise multiply of matrix1 with row_vector.
                **/
                shared_mat eltmul_broadcast_rowwise(shared_mat, shared_mat);
                /**
                Element Multiplication Rowwise
                ------------------------------

                The more general case is the element wise multiplication of two
                matrices A and B, with B transposed:

                > out = A * B^T

                Inputs
                ------

                shared_mat matrix1    : the matrix to multiply
                shared_mat matrix2    : the matrix to multiply after transposing

                Outputs
                -------

                shared_mat out : the element wise product of matrix1 and matrix2^T

                **/
                shared_mat eltmul_rowwise(shared_mat, shared_mat);
                shared_mat mul_with_bias(shared_mat, shared_mat, shared_mat);
                // operation of the form (A * x + B * y) + C, called with mul_add_mul_with_bias(A, x, B, y, C)
                shared_mat mul_add_mul_with_bias(shared_mat, shared_mat, shared_mat, shared_mat, shared_mat);
                shared_mat mul_add_mul_with_bias(std::initializer_list<shared_mat>);
                shared_mat mul_add_mul_with_bias(const std::vector<shared_mat>&);
                // operation of the form (A * x + B * y) + C, called with mul_add_mul_with_bias(A, x, B, y, C)
                // and with caveat that x is actually a column, and should be broadcasted
                shared_mat mul_add_broadcast_mul_with_bias(shared_mat, shared_mat, shared_mat, shared_mat, shared_mat);
                shared_mat add_broadcast(shared_mat, shared_mat);
                /**
                Graph<T>::add
                -------------

                Add a 2 matrices together. Broadcasts the sum if
                one of them is actually a vector (number of
                columns = d = 1)

                Inputs
                ------

                std::shared_ptr<Mat<T>> matrix1 : matrix to add
                std::shared_ptr<Mat<T>> matrix2 : matrix to add

                Outputs
                -------

                std::shared_ptr<Mat<T>> out : the sum of the matrices

                **/
                shared_mat add(shared_mat, shared_mat);
                shared_mat sub(shared_mat, shared_mat);
                shared_mat sub_broadcast(shared_mat, shared_mat);
                shared_mat sub_broadcast_reversed(shared_mat, shared_mat);


                /**
                Graph<T>::add
                -------------

                Add a list of matrices together, but does not perform any
                broadcasting (yet)

                Inputs
                ------

                std::initializer_list<std::shared_ptr<Mat<T>>> matrices : matrices to add

                Outputs
                -------

                std::shared_ptr<Mat<T>> out : the sum of the matrices
                **/
                shared_mat add(std::initializer_list<shared_mat>);
                shared_mat square(shared_mat);
                /**
                Graph<T>::sum
                -------------

                Sum the elements of a matrix into a 1x1 matrix.

                Inputs
                ------

                std::shared_ptr<Mat<T>> matrix1 : matrix to sum

                Outputs
                -------

                std::shared_ptr<Mat<T>> out : matrix sum

                **/
                shared_mat sum(shared_mat);
                /**
                Graph<T>::mean
                -------------

                Average the elements of a matrix into a 1x1 matrix.

                Inputs
                ------

                std::shared_ptr<Mat<T>> matrix1 : matrix to average

                Outputs
                -------

                std::shared_ptr<Mat<T>> out : matrix average

                **/
                shared_mat mean(shared_mat);
                shared_mat log(shared_mat);
                shared_mat exp(shared_mat);
                shared_mat cross_entropy(shared_mat, uint answer_idx);
                shared_mat binary_cross_entropy(shared_mat, T);
                shared_mat hstack(shared_mat, shared_mat);
                shared_mat hstack(std::initializer_list<shared_mat>);
                shared_mat hstack(const std::vector<shared_mat>&);
                shared_mat vstack(shared_mat, shared_mat);
                shared_mat vstack(std::initializer_list<shared_mat>);
                shared_mat vstack(const std::vector<shared_mat>&);
                shared_mat sigmoid(shared_mat);
                shared_mat softmax(shared_mat, T temperature=1.0);
                shared_mat steep_sigmoid(shared_mat matrix, T aggressiveness = 3.75);
                shared_mat transpose(shared_mat);
                shared_mat tanh(shared_mat);
                shared_mat relu(shared_mat);
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

                shared_mat matrix : where to apply the noise
                      T drop_prob : likelihood that an element of the matrix
                                    goes to 0

                Outputs
                -------

                shared_mat out : noisy matrix

                **/
                shared_mat dropout(shared_mat, T);
                shared_mat dropout_normalized(shared_mat, T);

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

                shared_mat matrix : where to apply the noise

                Outputs
                -------

                shared_mat out : noisy matrix

                **/
                shared_mat fast_dropout(shared_mat);
                shared_mat mul(shared_mat, shared_mat);
                shared_mat rows_pluck(shared_mat, Indexing::Index);
                shared_mat rows_cols_pluck(shared_mat, Indexing::Index, Indexing::Index);
                shared_mat row_pluck(shared_mat, int);
};


template<typename T>
std::ostream& operator<<(std::ostream&, const Graph<T>&);

#endif
