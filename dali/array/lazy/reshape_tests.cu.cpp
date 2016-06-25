#include <gtest/gtest.h>

#include "dali/array/array.h"
#include "dali/runtime_config.h"
#include "dali/utils/print_utils.h"
#include "dali/array/functor.h"
#include "dali/array/lazy_op.h"

TEST(ArrayReshapeTests, take_assign) {
    auto train_y = Array::ones({10}, DTYPE_INT32);

    int num_selected = 5;
    auto idxes = Array::ones({num_selected}, DTYPE_INT32);

    auto src = Array::ones({num_selected}, DTYPE_INT32);

    train_y[idxes] += src;
}

TEST(ArrayReshapeTests, rows_pluck_forward_correctness_1d_source) {
    const int num_plucks = 4;
    Array A({10}, DTYPE_FLOAT);
    A = initializer::uniform(-20.0, 20.0);
    Array indices({num_plucks, 1, 1}, DTYPE_INT32);
    indices = initializer::uniform(0, A.shape()[0] - 1);

    Array res = A[indices];

    EXPECT_EQ(std::vector<int>({num_plucks, 1, 1}), res.shape());

    #ifdef DALI_USE_CUDA
        EXPECT_TRUE(res.memory()->is_fresh(memory::Device::gpu(0)));
    #endif

    for (int pluck_idx = 0; pluck_idx < indices.number_of_elements(); ++pluck_idx) {
        auto actual_el = res[pluck_idx][0][0];
        auto expected_el = A[(int)indices(pluck_idx)];
        EXPECT_NEAR(actual_el, expected_el, 1e-4);
    }
}

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


TEST(ArrayReshapeTests, take_from_rows_2D) {
    Array A({4, 3}, DTYPE_INT32);
    A = initializer::uniform(-20, 20.0);

    Array indices({4}, DTYPE_INT32);
    indices = initializer::uniform(0, A.shape()[1] - 1);

    Array res = lazy::take_from_rows(A, indices);

    for (int pluck_idx = 0; pluck_idx < indices.number_of_elements(); ++pluck_idx) {
        auto col_idx      = (int)indices(pluck_idx);
        auto actual_elt   = (int)res(pluck_idx);
        auto expected_elt = (int)A[pluck_idx][col_idx];
        EXPECT_EQ(expected_elt, actual_elt);
    }
}

TEST(ArrayReshapeTests, take_from_rows_3D) {
    Array A({2, 3, 4}, DTYPE_INT32);
    A = initializer::uniform(-20, 20.0);

    Array indices({2, 3}, DTYPE_INT32);
    indices = initializer::uniform(0, A.shape()[2] - 1);

    Array res = lazy::take_from_rows(A, indices);

    for (int dim1 = 0; dim1 < A.shape()[0]; ++dim1) {
        for (int dim2 = 0; dim2 < A.shape()[1]; ++dim2) {
            auto col_idx      = (int)indices[dim1][dim2];
            auto actual_elt   = (int)res[dim1][dim2];
            auto expected_elt = (int)A[dim1][dim2][col_idx];
            EXPECT_EQ(expected_elt, actual_elt);
        }
    }
}

TEST(ArrayReshapeTests, take_from_rows_assign) {
    Array x({2, 4}, DTYPE_INT32);
    x = initializer::arange();


    Array indices({2}, DTYPE_INT32);
    indices[0] = 0;
    indices[1] = 1;

    auto x_view = x.take_from_rows(indices);
    x_view = 36;

    EXPECT_EQ(36, (int)x[0][0]);
    EXPECT_EQ(1,  (int)x[0][1]);
    EXPECT_EQ(2,  (int)x[0][2]);
    EXPECT_EQ(3,  (int)x[0][3]);

    EXPECT_EQ(4,  (int)x[1][0]);
    EXPECT_EQ(36, (int)x[1][1]);
    EXPECT_EQ(6,  (int)x[1][2]);
    EXPECT_EQ(7,  (int)x[1][3]);
}


TEST(ArrayReshapeTests, take_from_rows_assign_lazy) {
    auto x = Array::zeros({2, 4}, DTYPE_INT32);
    x = initializer::arange();

    Array y = Array({2}, DTYPE_INT32);
    y[0] = 44;
    y[1] = 31;

    Array indices({2}, DTYPE_INT32);
    indices[0] = 0;
    indices[1] = 1;

    ArraySubtensor x_view = x.take_from_rows(indices);
    x_view = y + 2;

    EXPECT_EQ(46, (int)x[0][0]);
    EXPECT_EQ(1,  (int)x[0][1]);
    EXPECT_EQ(2,  (int)x[0][2]);
    EXPECT_EQ(3,  (int)x[0][3]);

    EXPECT_EQ(4,  (int)x[1][0]);
    EXPECT_EQ(33, (int)x[1][1]);
    EXPECT_EQ(6,  (int)x[1][2]);
    EXPECT_EQ(7,  (int)x[1][3]);
}


TEST(ArrayReshapeTests, gather_assign_lazy) {
    auto x = Array::zeros({4, 2}, DTYPE_INT32);
    x = initializer::arange();

    Array y = Array({3, 2}, DTYPE_INT32);
    y = std::vector<std::vector<int>> {
        {10,   20},
        {300,  400},
        {5000, 6000}
    };

    Array indices({3}, DTYPE_INT32);
    indices[0] = 0;
    indices[1] = 1;
    indices[2] = 0;

    auto x_view = x[indices];

    x_view += y + 2;

    EXPECT_EQ(std::vector<int>({3,2}), x_view.shape());

    EXPECT_EQ(0 + 10  + 5000 + 2 + 2, (int)x[0][0]);
    EXPECT_EQ(1 + 20  + 6000 + 2 + 2, (int)x[0][1]);
    EXPECT_EQ(2 + 300 + 2,            (int)x[1][0]);
    EXPECT_EQ(3 + 400 + 2,            (int)x[1][1]);

    EXPECT_EQ(4,  (int)x[2][0]);
    EXPECT_EQ(5,  (int)x[2][1]);
    EXPECT_EQ(6,  (int)x[3][0]);
    EXPECT_EQ(7,  (int)x[3][1]);
}

TEST(ArrayReshapeTests, DISABLED_gather_assign_basic) {
    auto x = Array::zeros({4}, DTYPE_INT32);
    auto indices = Array::ones({2, 3}, DTYPE_INT32);

    Array subset;

    subset = x[indices];
    subset.print();
}

TEST(ArrayReshapeTests, DISABLED_gather_assign_advanced) {
    auto x = Array::zeros({2, 2, 4, 2}, DTYPE_INT32);
    x = initializer::arange();

    Array y = Array({2, 4, 2}, DTYPE_INT32);
    y = std::vector<std::vector<std::vector<int>>> {
        {
            {10,       20},
            {300,     400},
            {5000,   6000},
            {70000, 80000},
        },
        {
            {15,       25},
            {305,     405},
            {5005,   6005},
            {70005, 80005},
        }
    };

    Array indices({4, 5}, DTYPE_INT32);
    ((Array)indices[Slice(0, 2)]) = 0;
    ((Array)indices[Slice(2, 4)]) = 1;

    auto x_view = x[indices];
    EXPECT_EQ(std::vector<int>({4, 5, 2, 4, 2}), x_view.shape());

    std::cout << "hello" << std::endl;

    Array blah = x_view;

    blah.print();

    // x_view.print();

    std::cout << "hello" << std::endl;

    y.print();

    x_view += (Array)y[Broadcast()][Broadcast()];
    x_view.print();

    EXPECT_TRUE(Array::equals(x[0][0], 10 * y));
    EXPECT_TRUE(Array::equals(x[1][0], 10 * y));
    EXPECT_TRUE(Array::equals(x[1][0], 10 * y));
    EXPECT_TRUE(Array::equals(x[1][1], 10 * y));
}
