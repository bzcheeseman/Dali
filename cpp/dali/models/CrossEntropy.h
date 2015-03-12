#ifndef CROSSENTROPY_MAT_H
#define CROSSENTROPY_MAT_H

#include "dali/models/Softmax.h"

template<typename T>                         T cross_entropy(Mat<T>, uint&);
template<typename T, typename M>             T cross_entropy(Mat<T>, const M);


/**
Masked Cross Entropy Loss
-------------------------

Given a probability distribution p at a time T, for k channels,
and k different targets, apply KL Divergence loss
on the channels that are have T >= loss_start[k] and
T < loss_start[k] + codelens[k] (so from T to T+codelen error
will be picked up by channel k).

Inputs
------

Mat<Z> logprobs : the log probabilities (unnormalized)
uint& T : the log probabilities (unnormalized)
shared_eigen_index_vector loss_start : where to start picking up errors for channel k
shared_eigen_index_vector codelens : how long does channel k pick up errors
const M targets : the labels at time T

Outputs
-------

Z cost : the total KL divergence at this time step for
         the relevant channels
*/
template<typename T, typename M>             T masked_cross_entropy(Mat<T>, uint&, int,  int,  const M);
template<typename T, typename M>             T masked_cross_entropy(Mat<T>, uint&, uint, uint, const M);
template<typename T, typename M>             T masked_cross_entropy(Mat<T>, uint&, int,  uint, const M);
template<typename T, typename M>             T masked_cross_entropy(Mat<T>, uint&, uint, int,  const M);
template<typename T, typename M>             T masked_cross_entropy(Mat<T>, uint&, shared_eigen_index_vector, shared_eigen_index_vector, const M);
template<typename T, typename M>             T masked_cross_entropy(Mat<T>, uint&, shared_eigen_index_vector, int, const M);
template<typename T, typename M>             T masked_cross_entropy(Mat<T>, uint&, shared_eigen_index_vector, uint, const M);
template<typename T, typename M>             T masked_cross_entropy(Mat<T>, uint&, int, shared_eigen_index_vector, const M);
template<typename T, typename M>             T masked_cross_entropy(Mat<T>, uint&, uint, shared_eigen_index_vector, const M);
/**
Masked Cross Entropy Loss
-------------------------

(does not calculate gradient, only reports error)

See `masked_cross_entropy`
*/
template<typename T, typename M>             T masked_cross_entropy_no_grad(Mat<T>, uint&, int,  int,  const M);
template<typename T, typename M>             T masked_cross_entropy_no_grad(Mat<T>, uint&, uint, uint, const M);
template<typename T, typename M>             T masked_cross_entropy_no_grad(Mat<T>, uint&, int,  uint, const M);
template<typename T, typename M>             T masked_cross_entropy_no_grad(Mat<T>, uint&, uint, int,  const M);
template<typename T, typename M>             T masked_cross_entropy_no_grad(Mat<T>, uint&, shared_eigen_index_vector, shared_eigen_index_vector, const M);
template<typename T, typename M>             T masked_cross_entropy_no_grad(Mat<T>, uint&, shared_eigen_index_vector, int, const M);
template<typename T, typename M>             T masked_cross_entropy_no_grad(Mat<T>, uint&, shared_eigen_index_vector, uint, const M);
template<typename T, typename M>             T masked_cross_entropy_no_grad(Mat<T>, uint&, int, shared_eigen_index_vector, const M);
template<typename T, typename M>             T masked_cross_entropy_no_grad(Mat<T>, uint&, uint, shared_eigen_index_vector, const M);
/**
Masked Sum
----------

Sum values[k] if timestep T
if within [loss_start[k], loss_start[k] + codelens[k]),
gradient is columnwise vector of 1s.

Inputs:
-------

Mat<Z> values : the data columns subject to summing.
uint& T : the log probabilities (unnormalized)
shared_eigen_index_vector loss_start : where to start picking up errors for channel k
shared_eigen_index_vector codelens : how long does channel k pick up errors

Outputs:
--------

Z cost : the total sum along the non-masked columns of values.
*/
template<typename T>                         T masked_sum(Mat<T>, uint&, uint, uint, const T&);
template<typename T>                         T masked_sum(Mat<T>, uint&, int,  int, const T&);
template<typename T>                         T masked_sum(Mat<T>, uint&, uint, int, const T&);
template<typename T>                         T masked_sum(Mat<T>, uint&, int,  uint, const T&);
template<typename T>                         T masked_sum(Mat<T>, uint&, int,  shared_eigen_index_vector, const T&);
template<typename T>                         T masked_sum(Mat<T>, uint&, uint, shared_eigen_index_vector, const T&);
template<typename T>                         T masked_sum(Mat<T>, uint&, shared_eigen_index_vector, uint, const T&);
template<typename T>                         T masked_sum(Mat<T>, uint&, shared_eigen_index_vector, int, const T&);
template<typename T>                         T masked_sum(Mat<T>, uint&, shared_eigen_index_vector, shared_eigen_index_vector, const T&);

/**
Masked Sum No Grad
------------------

(does not calculate gradient, only reports error)

See `masked_sum`
*/
template<typename T>                         T masked_sum_no_grad(Mat<T>, uint&, uint, uint, const T&);
template<typename T>                         T masked_sum_no_grad(Mat<T>, uint&, int,  int, const T&);
template<typename T>                         T masked_sum_no_grad(Mat<T>, uint&, uint, int, const T&);
template<typename T>                         T masked_sum_no_grad(Mat<T>, uint&, int,  uint, const T&);
template<typename T>                         T masked_sum_no_grad(Mat<T>, uint&, int,  shared_eigen_index_vector, const T&);
template<typename T>                         T masked_sum_no_grad(Mat<T>, uint&, uint, shared_eigen_index_vector, const T&);
template<typename T>                         T masked_sum_no_grad(Mat<T>, uint&, shared_eigen_index_vector, uint, const T&);
template<typename T>                         T masked_sum_no_grad(Mat<T>, uint&, shared_eigen_index_vector, int, const T&);
template<typename T>                         T masked_sum_no_grad(Mat<T>, uint&, shared_eigen_index_vector, shared_eigen_index_vector, const T&);

#endif
