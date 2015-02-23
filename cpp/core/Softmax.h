#ifndef SOFTMAX_MAT_H
#define SOFTMAX_MAT_H

#include "Mat.h"

template<typename T> std::shared_ptr<Mat<T>> softmax(const std::shared_ptr<Mat<T>>);
template<typename T> std::shared_ptr<Mat<T>> softmax_transpose(const std::shared_ptr<Mat<T>>);

#endif