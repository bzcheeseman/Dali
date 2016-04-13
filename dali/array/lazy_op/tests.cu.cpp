#include <gtest/gtest.h>

#include "dali/utils/print_utils.h"

#include "dali/runtime_config.h"

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
    EXPECT_EQ(Evaluator<decltype(fake_expression)>::deduce_computation_device(fake(0), fake_expression), memory::Device::fake(0));

    // if everybody prefers the same device, return it.
    auto fake_expression2 = fake(4) * fake(4) + fake(4) + 1;
    EXPECT_EQ(Evaluator<decltype(fake_expression2)>::deduce_computation_device(fake(4), fake_expression2), memory::Device::fake(4));

    memory::WithDevicePreference device_prefence(memory::Device::fake(6));

    // if everybody prefereces differ fall back to default_preferred_device
    auto fake_expression3 = fake(2) * fake(4) + fake(5) + 1;
    EXPECT_EQ(Evaluator<decltype(fake_expression3)>::deduce_computation_device(fake(4), fake_expression3), memory::Device::fake(6));

    // if the memory was not allocated yet and we have a single argument we fall back to preffered device
    EXPECT_EQ(Evaluator<int>::deduce_computation_device(fake(1), 16), memory::Device::fake(1));

    memory::debug::fake_device_memories[10].fresh = true;



    EXPECT_EQ(Evaluator<int>::deduce_computation_device(fake(1), 16), memory::Device::fake(10));
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

    // res.memory()->print_debug_info({memory::Device::cpu(), memory::Device::gpu(0)}, true, x.dtype());
    EXPECT_EQ((float)(res(0)), 2 * 3  * 5);
    EXPECT_EQ((float)(res(1)), 7 * 11 * 13);
}

TEST(ArrayLazyOpsTests, lazy_shape_deduction) {
    Array x({16});
    Array y({16});
    Array z({16});


    // test deduction of shape of lazy expression
    auto partial = x * (y * z);
    EXPECT_EQ(partial.shape(), std::vector<int>({16}));

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
