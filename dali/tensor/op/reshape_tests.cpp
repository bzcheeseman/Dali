#include <chrono>
#include <vector>
#include <iomanip>
#include <functional>
#include <gtest/gtest.h>

#include "dali/test_utils.h"
#include "dali/tensor/tensor.h"
#include "dali/tensor/op/reshape.h"

using std::vector;

typedef MemorySafeTest TensorReshapeTests;

TEST_F(TensorReshapeTests, DISABLED_gather) {
    EXPERIMENT_REPEAT {
        auto A = Tensor::uniform(-20.0, 20.0, {5, 4});
        auto B = Tensor::uniform(0, A.shape()[1] - 1, {7});

        auto functor = [&](vector<Tensor> Xs)-> Tensor {
            return tensor_ops::gather(A, B);
        };
        ASSERT_TRUE(gradient_same(functor, {A}, 1e-4));
    }
}
