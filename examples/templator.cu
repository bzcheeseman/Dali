#include <iostream>
#include <vector>
#include "mshadow/tensor.h"
#include "dali/tensor/Mat.h"
#include "dali/math/ThrustSoftmaxTranspose.h"

#include <thrust/device_vector.h>
#include <thrust/equal.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>

using namespace mshadow;
using namespace mshadow::expr;
using std::vector;

void Print1DTensor(Tensor<cpu, 1, float> const &ts) {
  for (index_t i = 0; i < ts.size(0); ++i) {
    printf("%.2f ", ts[i]);
  }
  printf("\n");
}

void Print2DTensor(Tensor<cpu, 2, float> const &ts) {
    for (index_t i = 0; i < ts.size(0); ++i) {
        Print1DTensor(ts[i]);
    }
}

typedef float R;

int main() {
    dali_init();
    Mat<R> A(3, 4, weights<R>::uniform(10.0));
    A.w().print();
    TensorOps::softmax(A.dw().mutable_gpu_data(), A.w().gpu_data());
    A.dw().print();
}
