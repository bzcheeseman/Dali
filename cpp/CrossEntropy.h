#ifndef CROSSENTROPY_MAT_H
#define CROSSENTROPY_MAT_H

#include "Softmax.h"

template<typename T> T cross_entropy(std::shared_ptr<Mat<T>>, int&);
template<typename T, typename M> T cross_entropy(std::shared_ptr<Mat<T>>, const M);

template<typename T, typename M, typename K, typename F> T masked_cross_entropy(std::shared_ptr<Mat<T>>, uint&, const K&, const F&, const M);
template<typename T, typename M, typename K> T masked_cross_entropy(std::shared_ptr<Mat<T>>, uint&, const K&, shared_eigen_index_vector, const M);
template<typename T, typename M, typename F> T masked_cross_entropy(std::shared_ptr<Mat<T>>, uint&, shared_eigen_index_vector, const F&, const M);
template<typename T, typename M> T masked_cross_entropy(std::shared_ptr<Mat<T>>, uint&, shared_eigen_index_vector, shared_eigen_index_vector, const M);

template<typename T, typename K, typename F> T masked_sum(std::shared_ptr<Mat<T>>, uint&, const K&, const F&, T);
template<typename T, typename K> T masked_sum(std::shared_ptr<Mat<T>>, uint&, const K&, shared_eigen_index_vector, T);
template<typename T, typename F> T masked_sum(std::shared_ptr<Mat<T>>, uint&, shared_eigen_index_vector, const F&, T);
template<typename T> T masked_sum(std::shared_ptr<Mat<T>>, uint&, shared_eigen_index_vector, shared_eigen_index_vector, T);

template<typename T> T masked_sum(std::shared_ptr<Mat<T>>, uint&, int, shared_eigen_index_vector, T);
template<typename T, typename F> T masked_sum(std::shared_ptr<Mat<T>>, uint&, int, const F&, T);

#endif