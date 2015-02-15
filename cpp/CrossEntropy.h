#ifndef CROSSENTROPY_MAT_H
#define CROSSENTROPY_MAT_H

#include "Softmax.h"

template<typename T>                         T cross_entropy(std::shared_ptr<Mat<T>>, int&);
template<typename T, typename M>             T cross_entropy(std::shared_ptr<Mat<T>>, const M);
template<typename T, typename M>             T masked_cross_entropy(std::shared_ptr<Mat<T>>, uint&, int,  int,  const M);
template<typename T, typename M>             T masked_cross_entropy(std::shared_ptr<Mat<T>>, uint&, uint, uint, const M);
template<typename T, typename M>             T masked_cross_entropy(std::shared_ptr<Mat<T>>, uint&, int,  uint, const M);
template<typename T, typename M>             T masked_cross_entropy(std::shared_ptr<Mat<T>>, uint&, uint, int,  const M);
template<typename T, typename M>             T masked_cross_entropy(std::shared_ptr<Mat<T>>, uint&, shared_eigen_index_vector, shared_eigen_index_vector, const M);
template<typename T, typename M>             T masked_cross_entropy(std::shared_ptr<Mat<T>>, uint&, shared_eigen_index_vector, int, const M);
template<typename T, typename M>             T masked_cross_entropy(std::shared_ptr<Mat<T>>, uint&, shared_eigen_index_vector, uint, const M);
template<typename T, typename M>             T masked_cross_entropy(std::shared_ptr<Mat<T>>, uint&, int, shared_eigen_index_vector, const M);
template<typename T, typename M>             T masked_cross_entropy(std::shared_ptr<Mat<T>>, uint&, uint, shared_eigen_index_vector, const M);

template<typename T>                         T masked_sum(std::shared_ptr<Mat<T>>, uint&, uint, uint, const T&);
template<typename T>                         T masked_sum(std::shared_ptr<Mat<T>>, uint&, int,  int, const T&);
template<typename T>                         T masked_sum(std::shared_ptr<Mat<T>>, uint&, uint, int, const T&);
template<typename T>                         T masked_sum(std::shared_ptr<Mat<T>>, uint&, int,  uint, const T&);
template<typename T>                         T masked_sum(std::shared_ptr<Mat<T>>, uint&, int,  shared_eigen_index_vector, const T&);
template<typename T>                         T masked_sum(std::shared_ptr<Mat<T>>, uint&, uint, shared_eigen_index_vector, const T&);
template<typename T>                         T masked_sum(std::shared_ptr<Mat<T>>, uint&, shared_eigen_index_vector, uint, const T&);
template<typename T>                         T masked_sum(std::shared_ptr<Mat<T>>, uint&, shared_eigen_index_vector, int, const T&);
template<typename T>                         T masked_sum(std::shared_ptr<Mat<T>>, uint&, shared_eigen_index_vector, shared_eigen_index_vector, const T&);

#endif