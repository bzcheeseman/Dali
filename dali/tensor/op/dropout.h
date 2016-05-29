#ifndef DALI_TENSOR_OP_DROPOUT_H
#define DALI_TENSOR_OP_DROPOUT_H

#include "dali/tensor/tensor.h"

namespace tensor_ops {
    /*
     * Dropout and Dropout Unnormalized
     * ================================
     *
     * Apply bernoulli noise to a matrix (e.g. to regularize a
     * neural network). Randomly sets elements in the input
     * tensor to 0.
     *
     * When normalized:
     * ----------------
     * When normalized the non-zeroed out elements
     * are scaled up by `alpha = 1.0 / (1 - drop_prob)` to preserve
     * the same distributional statistics.
     *
     * When unnormalized:
     * ------------------
     * Elements that were not dropped are kept the same.
     * In this approach the network is trained with a noise distribution
     * during training, and typically during inference (test time)
     * dropout is switched off (drop_prob = 0), and the units are
     * multiplied by `alpha = 1 - drop_prob` to recover the same
     * distributional statistics. Since this is not automatic, the
     * user must correct for this change themselves.
     *
     * Paper Abstract
     * --------------
     * Deep neural nets with a large number of parameters are
     * very powerful machine learning systems. However, overfitting
     * is a serious problem in such networks. Large networks are
     * also slow to use, making it difficult to deal with overfitting
     * by combining the predictions of many different large neural
     * nets at test time. Dropout is a technique for addressing
     * this problem. The key idea is to randomly drop units (along
     * with their connections) from the neural network during training.
     * This prevents units from co-adapting too much. During training,
     * dropout samples from an exponential number of different
     * “thinned” networks. At test time, it is easy to approximate
     * the effect of averaging the predictions of all these thinned
     * networks by simply using a single unthinned network that has
     * smaller weights. This significantly reduces overfitting and
     * gives major improvements over other regularization methods.
     *
     * - Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky,
     * Ilya Sutskever, Ruslan Salakhutdinov, "Dropout: A Simple Way
     * to Prevent Neural Networks from Overfitting," JMLR 2014
     *
     * Args:
     *   drop_prob : likelihood that an element of the matrix goes to 0
     */
    Tensor dropout(const Tensor&, const double& drop_prob);
    Tensor dropout_unnormalized(const Tensor&, const double& drop_prob);
    /*
     * Fast Dropout
     * ============
     *
     * Apply Gaussian Noise a standard deviation of 1 and a
     * mean of 1 to a matrix (e.g. to regularize it)
     *
     * Paper Abstract
     * --------------
     * Preventing feature co-adaptation by encouraging independent
     * contributions from differ- ent features often improves
     * classification and regression performance. Dropout training
     * (Hinton et al., 2012) does this by randomly dropping out
     * (zeroing) hidden units and input features during training
     * of neural networks. However, repeatedly sampling a random
     * subset of input features makes training much slower. Based
     * on an examination of the implied objective function of dropout
     * training, we show how to do fast dropout training by sampling
     * from or integrating a Gaussian approximation, instead of
     * doing Monte Carlo optimization of this objective. This
     * approximation, justified by the central limit theorem and
     * empirical evidence, gives an order of magnitude speedup and
     * more stability.
     *
     * - Sida I. Wang, Christopher D. Manning, "Fast dropout training",
     * ICML 2013
     *
     * Relevant reading:
     * https://gist.github.com/SnippyHolloW/8a0f820261926e2f41cc
     */
    Tensor fast_dropout(const Tensor&);
}

#endif
