#include <gtest/gtest.h>

#include "dali/utils/print_utils.h"
#include "dali/array/test_utils.h"
#include "dali/runtime_config.h"
#define DALI_USE_LAZY 0
#include "dali/array/op.h"
#include "dali/array/op/initializer.h"

using namespace op;


TEST(ArrayOtherTests, all_close_and_all_equals) {
    Array a({10,3,1,2}, DTYPE_DOUBLE);
    Array b({10,3,1,2}, DTYPE_DOUBLE);
    Array perturb({10,3,1,2}, DTYPE_DOUBLE);
    a       = initializer::uniform(-20.0, 20.0);
    b       = a + 200.0;
    perturb = initializer::uniform(-0.1, 0.1);

    EXPECT_TRUE(Array::equals(a, a));
    EXPECT_TRUE(Array::equals(b, b));
    EXPECT_TRUE(Array::allclose(b, b, 1e-5));
    EXPECT_TRUE(Array::allclose(b, b, 1e-5));

    EXPECT_FALSE(Array::equals(a ,b));
    EXPECT_FALSE(Array::allclose(a, b, 1e-5));


    EXPECT_FALSE(Array::equals(a, a + perturb));
    EXPECT_TRUE(Array::allclose(a, a + perturb, 0.11));
}

TEST(ArrayOtherTests, argsort) {
    // shape of matrix has no influence on
    // argsort
    Array A({2, 3, 4, 3});
    A = initializer::uniform(-20.0, 20.0);

    Array res = A.argsort();

    double lowest = std::numeric_limits<double>::lowest();

    for (int i = 0; i < res.number_of_elements(); i++) {
        EXPECT_TRUE((double)A((int)res(i)) + 1e-6 >= lowest);
        lowest = A((int)res(i));
    }

    for (int axis = 0; axis < A.ndim(); axis++) {
        // put argsort dimension last
        auto res_axis = ((Array)A.argsort(axis));
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

TEST(ArrayOtherTests, argsort_strided) {
    // shape of matrix has no influence on
    // argsort
    Array A({2, 3, 4, 3});

    A = initializer::uniform(-20.0, 20.0);

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
        auto res_axis = ((Array)A.argsort(axis));
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
