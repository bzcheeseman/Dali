#include <chrono>
#include <vector>
#include <iomanip>
#include <functional>
#include <gtest/gtest.h>

#include "dali/test_utils.h"
#include "dali/tensor/tensor.h"
#include "dali/tensor/op/reshape.h"
#include "dali/array/op/initializer.h"

using std::vector;

TEST(TensorReshapeTests, gather) {
    EXPERIMENT_REPEAT {
        auto A = Tensor::uniform(-20.0, 20.0, {5, 4}, DTYPE_DOUBLE);
        auto B = Tensor::uniform(0, A.shape()[1] - 1, {7}, DTYPE_INT32);

        auto functor = [&](vector<Tensor> Xs)-> Tensor {
            return tensor_ops::gather(A, B);
        };
        ASSERT_TRUE(gradient_same(functor, {A}, 1e-4));
    }

}
