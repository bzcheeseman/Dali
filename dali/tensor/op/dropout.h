#ifndef DALI_TENSOR_OP_DROPOUT_H
#define DALI_TENSOR_OP_DROPOUT_H

#include <vector>

#include "dali/tensor/Mat.h"
#include "dali/tensor/Tape.h"
#include "dali/utils.h"

template<typename R> class Mat;

namespace matops {

    template<typename R>
    struct Dropout {
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

        static std::vector<Mat<R>> dropout(const std::vector<Mat<R>>&, R);
        static std::vector<Mat<R>> dropout_normalized(const std::vector<Mat<R>>&, R);

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
    };
}

#endif
