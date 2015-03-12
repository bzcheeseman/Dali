#ifndef SOFTMAX_MAT_H
#define SOFTMAX_MAT_H

#include "dali/core.h"

template<typename T> Mat<T> softmax(Mat<T>);
template<typename T> Mat<T> softmax_transpose(Mat<T>);

#endif
