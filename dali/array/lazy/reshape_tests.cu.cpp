#include <gtest/gtest.h>

#include "dali/array/array.h"
#include "dali/runtime_config.h"
#include "dali/utils/print_utils.h"
#include "dali/array/functor.h"

#define DALI_USE_LAZY 1
#include "dali/array/op.h"


TEST(ArrayReshapeTests, rows_pluck_forward_correctness) {
    const int num_plucks = 4;
    Array A({10, 5}, DTYPE_FLOAT);
    A = initializer::uniform(-20.0, 20.0);
    Array indices({num_plucks, 1, 1}, DTYPE_INT32);
    indices = initializer::uniform(0, A.shape()[0] - 1);

    Array res = A[indices];

    EXPECT_EQ(std::vector<int>({num_plucks, 1, 1, A.shape()[1]}), res.shape());

    #ifdef DALI_USE_CUDA
        EXPECT_TRUE(res.memory()->is_fresh(memory::Device::gpu(0)));
    #endif

    for (int pluck_idx = 0; pluck_idx < indices.number_of_elements(); ++pluck_idx) {
        auto actual_row = res[pluck_idx][0][0];
        auto expected_row = A[(int)indices(pluck_idx)];
        EXPECT_TRUE(Array::allclose(actual_row, expected_row, 1e-4));
    }
}


TEST(ArrayReshapeTests, rows_pluck_forward_correctness2) {
    Array A({3, 3, 3, 3}, DTYPE_FLOAT);

    A = initializer::uniform(-20.0, 20.0);
    Array indices({2}, DTYPE_INT32);
    indices = initializer::uniform(0, A.shape()[0] - 1);

    A = A.swapaxes(2,0);
    Array res = A[indices];

    for (int pluck_idx = 0; pluck_idx < indices.number_of_elements(); ++pluck_idx) {
        auto actual_row = res[pluck_idx];
        auto expected_row = A[(int)indices(pluck_idx)];
        EXPECT_TRUE(Array::allclose(actual_row, expected_row, 1e-4));
    }

}
