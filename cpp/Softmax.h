#ifndef SOFTMAX_MAT_H
#define SOFTMAX_MAT_H

#include "Mat.h"

template<typename T> std::shared_ptr<Mat<T>> softmax(std::shared_ptr<Mat<T>>);
template<typename T> std::shared_ptr<Mat<T>> softmax_transpose(std::shared_ptr<Mat<T>>);

#endif