#ifndef DALI_ARRAY_EXPRESSION_EYE_H
#define DALI_ARRAY_EXPRESSION_EYE_H

#include "dali/array/array.h"

namespace op {
Array eye(int size);
Array eye(int rows, int cols);
Array diag(Array diag);
Array diag(Array diag, int rows);
Array diag(Array diag, int rows, int cols);
}

#endif  // DALI_ARRAY_EXPRESSION_EYE_H
