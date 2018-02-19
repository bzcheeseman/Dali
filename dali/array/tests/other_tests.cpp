#include <gtest/gtest.h>
#include "dali/array/op/random.h"
#include "dali/array/op/top_k.h"
#include "dali/array/op/arange.h"

TEST(ArgsortTests, argsort) {
    // shape of matrix has no influence on argsort
    auto A = op::uniform(-20.0, 20.0, {2, 3});
    A.eval();
    Array res = A.argsort();
    double lowest = std::numeric_limits<double>::lowest();
    for (int i = 0; i < res.number_of_elements(); i++) {
        EXPECT_TRUE((double)A((int)res(i)) + 1e-6 >= lowest);
        lowest = A((int)res(i));
    }
    for (int axis = 0; axis < A.ndim(); axis++) {
        // put argsort dimension last
        auto res_axis = A.argsort(axis);
        res_axis = res_axis.swapaxes(axis, -1).reshape({-1, A.shape()[axis]});
        auto A_2d = A.swapaxes(axis, -1).reshape({-1, A.shape()[axis]});

        for (int i = 0; i < res_axis.shape()[0]; i++) {
            // ensure always increasing
            double lowest = std::numeric_limits<double>::lowest();
            for (int j = 0; j < res_axis.shape()[1]; j++) {
                EXPECT_TRUE((double)A_2d[i]((int)res_axis[i][j]) >= lowest);
                lowest = A_2d[i]((int)res_axis[i][j]);
            }
        }
    }
}

TEST(ArgsortTests, argsort_strided) {
    // shape of matrix has no influence on argsort
    auto A = op::uniform(-20.0, 20.0, {2, 3, 4, 3});
    A = A.swapaxes(1, 2);
    A = A[Slice(0, 2)][Slice(0, 4, -2)][Slice(0, 3)][Slice(0, 3)];

    Array res = A.argsort();

    double lowest = std::numeric_limits<double>::lowest();

    for (int i = 0; i < res.number_of_elements(); i++) {
        EXPECT_TRUE((double)A((int)res(i)) + 1e-6 >= lowest);
        lowest = A((int)res(i));
    }

    for (int axis = 0; axis < A.ndim(); axis++) {
        // put argsort dimension last
        auto res_axis = A.argsort(axis);
        res_axis = res_axis.swapaxes(axis, -1).reshape({A.number_of_elements() / A.shape()[axis], A.shape()[axis]});
        auto A_2d = A.swapaxes(axis, -1).reshape({A.number_of_elements() / A.shape()[axis], A.shape()[axis]});

        for (int i = 0; i < res_axis.shape()[0]; i++) {
            // ensure always increasing
            double lowest = std::numeric_limits<double>::lowest();
            for (int j = 0; j < res_axis.shape()[1]; j++) {
                EXPECT_TRUE((double)A_2d[i]((int)res_axis[i][j]) >= lowest);
                lowest = A_2d[i]((int)res_axis[i][j]);
            }
        }
    }
}
