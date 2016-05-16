#include <gtest/gtest.h>


#include "dali/array/function/lazy_evaluator.h"
#include "dali/runtime_config.h"
#include "dali/utils/print_utils.h"
#include "dali/array/functor.h"

#define DALI_USE_LAZY 1
#include "dali/array/op.h"


TEST(ArrayLazyOpsTests, imports) {
    ASSERT_TRUE(lazy::ops_loaded);
}

TEST(ArrayLazyOpsTests, lazy_device_deduction) {
    for (int i = 0; i < memory::debug::MAX_FAKE_DEVICES; ++i) {
        memory::debug::fake_device_memories[i].fresh = false;
    }


    memory::debug::enable_fake_devices = true;
    auto fake = [](int number) {
        return Array({16}, DTYPE_FLOAT, memory::Device::fake(number));
    };

    // if everybody prefers the same device, return it.
    auto fake_expression = fake(0) * fake(0) + fake(0) + 1;
    EXPECT_EQ(LazyEvaluator<decltype(fake_expression)>::deduce_computation_device(fake(0), fake_expression), memory::Device::fake(0));

    // if everybody prefers the same device, return it.
    auto fake_expression2 = fake(4) * fake(4) + fake(4) + 1;
    EXPECT_EQ(LazyEvaluator<decltype(fake_expression2)>::deduce_computation_device(fake(4), fake_expression2), memory::Device::fake(4));

    memory::WithDevicePreference device_prefence(memory::Device::fake(6));

    // if everybody prefereces differ fall back to default_preferred_device
    auto fake_expression3 = fake(2) * fake(4) + fake(5) + 1;
    EXPECT_EQ(LazyEvaluator<decltype(fake_expression3)>::deduce_computation_device(fake(4), fake_expression3), memory::Device::fake(6));

    // if the memory was not allocated yet and we have a single argument we fall back to preffered device
    EXPECT_EQ(LazyEvaluator<int>::deduce_computation_device(fake(1), 16), memory::Device::fake(1));

    memory::debug::fake_device_memories[10].fresh = true;



    EXPECT_EQ(LazyEvaluator<int>::deduce_computation_device(fake(1), 16), memory::Device::fake(10));
    memory::debug::fake_device_memories[10].fresh = false;

    memory::debug::enable_fake_devices = false;
}

TEST(ArrayLazyOpsTests, lazy_dtype_deduction) {
    Array x({16}, DTYPE_FLOAT);
    Array y({16}, DTYPE_FLOAT);
    Array z({16}, DTYPE_FLOAT);


    // test deduction of dtype of lazy expression
    auto partial = x * (y * z);
    EXPECT_EQ(partial.dtype(), DTYPE_FLOAT);

    // detect wrong dtype during lazy expression construction.
    Array z_wrong({16}, DTYPE_DOUBLE);
    EXPECT_THROW(x * (y * z_wrong), std::runtime_error);

    // assiging to preallocated memory
    Array q({16}, DTYPE_FLOAT);
    q = partial;

    // auto allocate memory
    Array qp = x * (y * z);

    // cannot assign to memory of wrong type
    Array q2({16}, DTYPE_INT32);
    EXPECT_THROW(q2 = x * (y * z), std::runtime_error);
}

TEST(ArrayLazyOpsTests, lazy_binary_correctness) {
    Array x({2,1});
    Array y({2,1});
    Array z({2,1});

    x(0) = 2; y(0) = 3;  z(0) = 5;
    x(1) = 7; y(1) = 11; z(1) = 13;

    auto partial = x * y * z;
    Array res = partial;

    EXPECT_EQ((float)(res(0)), 2 * 3  * 5);
    EXPECT_EQ((float)(res(1)), 7 * 11 * 13);
}

TEST(ArrayLazyOpsTests, elementwise_F) {
    auto x = Array::zeros({2,1});
    auto expr = lazy::F<functor::sigmoid>(x);
    Array y = expr;
    ASSERT_NEAR((float)y(0), 0.5, 1e-4);
    ASSERT_NEAR((float)y(1), 0.5, 1e-4);
}

// Test below passes, but stresses the template deduction
// of compiler, and takes much longer to compile than
// rest of code.
// TEST(ArrayLazyOpsTests, long_chain) {
//     Array x({2,1});
//     Array y({2,1});
//     Array z({2,1});
//
//     int lazy_evaluator_calls = 0;
//     auto callback_handle = debug::lazy_evaluation_callback.register_callback([&](const Array&) {
//         lazy_evaluator_calls += 1;
//     });
//     auto partial = (
//         lazy::sigmoid(lazy::tanh(x)) * 2 +
//         x * y * lazy::sign(z) * 2 +
//         1 +
//         x +
//         lazy::log_or_zero(y)
//     );
//
//     ASSERT_EQ(lazy_evaluator_calls, 0);
//     Array result = partial;
//     ASSERT_EQ(lazy_evaluator_calls, 1);
//     debug::lazy_evaluation_callback.deregister_callback(callback_handle);
// }

TEST(ArrayLazyOpsTests, sum_all) {
   auto z = Array::zeros({2,4}, DTYPE_FLOAT);
   auto o = Array::ones({2,4}, DTYPE_FLOAT);

   ASSERT_NEAR((float)(Array)z.sum(), 0.0, 1e-4);
   ASSERT_NEAR((float)(Array)o.sum(), 8.0, 1e-4);
}

// TODO(jonathan): make this test more gnarly
TEST(ArrayLazyOpsTests, argmax_axis) {
   auto z = Array::zeros({2,4}, DTYPE_FLOAT);

   z[0][1] = 2.0;
   z[1][3] = 3.0;
   Array max_values = lazy::max_axis(z, 1);
   Array max_indices = lazy::argmax_axis(z, 1);

   ASSERT_NEAR(2.0, (float)max_values[0], 1e-6);
   ASSERT_NEAR(3.0, (float)max_values[1], 1e-6);

   ASSERT_EQ(DTYPE_INT32, max_indices.dtype());

   ASSERT_EQ(1, (int)max_indices[0]);
   ASSERT_EQ(3, (int)max_indices[1]);
}

TEST(ArrayLazyOpsTests, argmin_axis) {
   auto z = Array::zeros({2,4}, DTYPE_FLOAT);
   z[0][1] = -2.0;
   z[1][3] = -3.0;
   Array min_values = lazy::min_axis(z, 1);
   Array min_indices = lazy::argmin_axis(z, 1);

   ASSERT_NEAR(-2.0, (float)min_values[0], 1e-6);
   ASSERT_NEAR(-3.0, (float)min_values[1], 1e-6);

   ASSERT_EQ(DTYPE_INT32, min_indices.dtype());

   ASSERT_EQ(1, (int)min_indices[0]);
   ASSERT_EQ(3, (int)min_indices[1]);
}



TEST(ArrayLazyOpsTests, lazy_shape_deduction) {
    Array x({16});
    Array y({16});
    Array z({16});

    // test deduction of shape of lazy expression
    auto partial = x * (y * z);
    EXPECT_EQ(partial.bshape(), std::vector<int>({16}));

    // detect wrong shape during lazy expression construction.
    Array z_wrong({20,4});
    EXPECT_THROW(x * (y * z_wrong), std::runtime_error);

    // assiging to preallocated memory
    Array q({16});
    q = partial;

    // auto allocate memory
    Array qp = x * (y * z);

    // cannot assign to memory of wrong shape
    Array q2({14, 5});
    EXPECT_THROW(q2 = x * (y * z), std::runtime_error);
}

TEST(ArrayLazyOpsTests, broadcasted_add) {
    auto out = Array::zeros({2,3,4}, DTYPE_INT32);
    auto A = Array::ones({2,3,4}, DTYPE_INT32);
    auto B = Array::ones({3},     DTYPE_INT32);

    B = B[Broadcast()][Slice(0,3)][Broadcast()];

    out = A +  2 * B;

    ASSERT_EQ((int)(Array)out.sum(), 2 * 3 * 4 * 3);
}

TEST(ArrayTests, sum_axis) {
    Array x = Array::ones({2,3,4}, DTYPE_INT32);
    for (int reduce_axis = 0; reduce_axis < x.ndim(); reduce_axis++) {
        Array y = lazy::sum_axis(x, reduce_axis);
        std::vector<int> expected_shape;
        switch (reduce_axis) {
            case 0:
                expected_shape = {3, 4};
                break;
            case 1:
                expected_shape = {2, 4};
                break;
            case 2:
                expected_shape = {2, 3};
                break;
        }
        EXPECT_EQ(expected_shape, y.shape());
        for (int i = 0; i < y.number_of_elements(); i++) {
            EXPECT_EQ(x.shape()[reduce_axis], (int)y(i));
        }
    }
}


TEST(ArrayTests, sum_axis_broadcasted) {
    Array x = Array::ones({2,3,4}, DTYPE_INT32)[Slice(0,2)][Slice(0,3)][Slice(0,4)][Broadcast()];

    for (int reduce_axis = 0; reduce_axis < 3; reduce_axis++) {
        Array y = lazy::sum_axis(x, reduce_axis);
        std::vector<int> expected_shape;
        switch (reduce_axis) {
            case 0:
                expected_shape = {3, 4, 1};
                break;
            case 1:
                expected_shape = {2, 4, 1};
                break;
            case 2:
                expected_shape = {2, 3, 1};
                break;
        }
        EXPECT_EQ(expected_shape, y.shape());

        for (int i = 0; i < y.number_of_elements(); i++) {
            EXPECT_EQ(x.shape()[reduce_axis], (int)y(i));
        }
    }
}

TEST(ArrayTests, sum_axis_broadcasted2) {
    Array x = Array::ones({2,4}, DTYPE_INT32)[Slice(0,2)][Broadcast()][Slice(0,4)];

    for (int reduce_axis = 0; reduce_axis < 3; reduce_axis++) {
        Array y = lazy::sum_axis(x, reduce_axis);
        std::vector<int> expected_bshape;
        switch (reduce_axis) {
            case 0:
                expected_bshape = {-1, 4};
                break;
            case 1:
                expected_bshape = {2, 4};
                break;
            case 2:
                expected_bshape = {2, -1};
                break;
        }
        EXPECT_EQ(expected_bshape, y.bshape());

        for (int i = 0; i < y.number_of_elements(); i++) {
            EXPECT_EQ(x.shape()[reduce_axis], (int)y(i));
        }
    }
}

TEST(ArrayTests, sum_axis_broadcasted_2D) {
    Array x = Array::ones({2,4}, DTYPE_INT32);

    for (int reduce_axis = 0; reduce_axis < 2; reduce_axis++) {
        Array y = lazy::sum_axis(x, reduce_axis);
        std::vector<int> expected_shape;
        switch (reduce_axis) {
            case 0:
                expected_shape = {4};
                break;
            case 1:
                expected_shape = {2};
                break;
        }
        EXPECT_EQ(expected_shape, y.bshape());

        for (int i = 0; i < y.number_of_elements(); i++) {
            EXPECT_EQ(x.shape()[reduce_axis], (int)y(i));
        }
    }
}

TEST(ArrayTests, sum_axis_broadcasted_1D) {
    Array x = Array::ones({6,}, DTYPE_INT32);

    EXPECT_THROW(lazy::sum_axis(x, 1), std::runtime_error);
    EXPECT_THROW(lazy::sum_axis(x, -1), std::runtime_error);

    Array y = lazy::sum_axis(x, 0);

    EXPECT_EQ(0, y.ndim());

    EXPECT_EQ(6, (int)y);
}
