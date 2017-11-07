// #include <gtest/gtest.h>

// #include "dali/array/array.h"
// #include "dali/runtime_config.h"
// #include "dali/utils/print_utils.h"
// #include "dali/array/functor.h"

// #define DALI_USE_LAZY 1
// #include "dali/array/op.h"


// TEST(ArrayLazyOpsTests, lazy_device_deduction) {
//     for (int i = 0; i < memory::debug::MAX_FAKE_DEVICES; ++i) {
//         memory::debug::fake_device_memories[i].fresh = false;
//     }


//     memory::debug::enable_fake_devices = true;
//     auto fake = [](int number) {
//         return Array({16}, DTYPE_FLOAT, memory::Device::fake(number));
//     };

//     // if everyone prefers the same device, use it.
//     auto fake_expression = fake(0) * fake(0) + fake(0) + 1;
//     typedef LazyEvaluator<Array,decltype(fake_expression)> fake_expr_eval1_t;
//     EXPECT_EQ(fake_expr_eval1_t::deduce_computation_device(fake(0), fake_expression), memory::Device::fake(0));

//     // if everyone prefers the same device, use it.
//     auto fake_expression2 = fake(4) * fake(4) + fake(4) + 1;

//     typedef LazyEvaluator<Array,decltype(fake_expression2)> fake_expr_eval2_t;
//     EXPECT_EQ(fake_expr_eval2_t::deduce_computation_device(fake(4), fake_expression2), memory::Device::fake(4));

//     memory::WithDevicePreference device_prefence(memory::Device::fake(6));

//     // if everyone's preferences differ then fall back to default_preferred_device:
//     auto fake_expression3 = fake(2) * fake(4) + fake(5) + 1;

//     typedef LazyEvaluator<Array,decltype(fake_expression3)> fake_expr_eval3_t;
//     EXPECT_EQ(fake_expr_eval3_t::deduce_computation_device(fake(4), fake_expression3), memory::Device::fake(6));

//     // if the memory was not allocated yet and we have a single argument we fall back to prefered device
//     EXPECT_EQ(ReduceOverArgs<DeviceReducer>::reduce(fake(1)), memory::Device::fake(1));
//     memory::debug::fake_device_memories[10].fresh = true;
//     EXPECT_EQ(ReduceOverArgs<DeviceReducer>::reduce(fake(1)), memory::Device::fake(10));
//     memory::debug::fake_device_memories[10].fresh = false;

//     memory::debug::enable_fake_devices = false;
// }

// TEST(ArrayLazyOpsTests, lazy_dtype_deduction) {
//     Array x({16}, DTYPE_FLOAT);
//     Array y({16}, DTYPE_FLOAT);
//     Array z({16}, DTYPE_FLOAT);


//     // test deduction of dtype of lazy expression
//     auto partial = x * (y * z);
//     EXPECT_EQ(partial.dtype(), DTYPE_FLOAT);

//     // detect wrong dtype during lazy expression construction.
//     Array z_wrong({16}, DTYPE_DOUBLE);
//     EXPECT_THROW(x * (y * z_wrong), std::runtime_error);

//     // assiging to preallocated memory
//     Array q({16}, DTYPE_FLOAT);
//     q = partial;

//     // auto allocate memory
//     Array qp = x * (y * z);

//     // cannot assign to memory of wrong type
//     Array q2({16}, DTYPE_INT32);
//     EXPECT_THROW(q2 = x * (y * z), std::runtime_error);
// }

// TEST(ArrayLazyOpsTests, lazy_shape_deduction) {
//     Array x({16});
//     Array y({16});
//     Array z({16});

//     // test deduction of shape of lazy expression
//     auto partial = x * (y * z);
//     EXPECT_EQ(partial.bshape(), std::vector<int>({16}));

//     // detect wrong shape during lazy expression construction.
//     Array z_wrong({20,4});
//     EXPECT_THROW(x * (y * z_wrong), std::runtime_error);

//     // assiging to preallocated memory
//     Array q({16});
//     q = partial;

//     // auto allocate memory
//     Array qp = x * (y * z);

//     // cannot assign to memory of wrong shape
//     Array q2({14, 5});
//     EXPECT_THROW(q2 = x * (y * z), std::runtime_error);
// }
