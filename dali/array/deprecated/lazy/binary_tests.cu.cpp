// #include <gtest/gtest.h>

// #include "dali/array/array.h"
// #include "dali/runtime_config.h"
// #include "dali/utils/print_utils.h"
// #include "dali/array/functor.h"

// #define DALI_USE_LAZY 1
// #include "dali/array/op.h"


// TEST(ArrayBinaryTests, lazy_binary_correctness) {
//     Array x({2,1});
//     Array y({2,1});
//     Array z({2,1});

//     x(0) = 2; y(0) = 3;  z(0) = 5;
//     x(1) = 7; y(1) = 11; z(1) = 13;

//     auto partial = x * y * z;
//     Array res = partial;

//     EXPECT_EQ((float)(res(0)), 2 * 3  * 5);
//     EXPECT_EQ((float)(res(1)), 7 * 11 * 13);
// }


// TEST(ArrayBinaryTests, broadcasted_add) {
//     auto out = Array::zeros({2,3,4}, DTYPE_INT32);
//     auto A = Array::ones({2,3,4}, DTYPE_INT32);
//     auto B = Array::ones({3},     DTYPE_INT32);

//     B = B[Broadcast()][Slice(0,3)][Broadcast()];

//     out = A +  2 * B;

//     ASSERT_EQ((int)(Array)out.sum(), 2 * 3 * 4 * 3);
// }

// TEST(ArrayBinaryTests, advanced_striding_with_reductions) {
//     Array x = Array::arange({3,4});
//     Array y = Array::arange({3,4});
//     y = y[Slice(0,3)][Slice(0,4,-1)];
//     for (int i =0; i <12; ++i) y(i) = i;

//     Array z =  lazy::sum(lazy::equals(x, y));
//     EXPECT_EQ(12, (int)z);
// }

// TEST(ArrayBinaryTests, advanced_striding_with_reductions1) {
//     Array x = Array::arange({3,4});
//     Array y = Array::arange({3,4});
//     y = y[Slice(0,3,-1)];
//     for (int i =0; i <12; ++i) y(i) = i;

//     Array z =  lazy::sum(lazy::equals(x, y));
//     EXPECT_EQ(12, (int)z);
// }

// TEST(ArrayBinaryTests, advanced_striding_with_reductions2) {
//     Array x = Array::arange({12});
//     Array y_source = Array::arange({12,2});
//     Array y = y_source[Slice(0,12)][1];

//     for (int i =0; i <12; ++i) y(i) = i;

//     Array z =  lazy::sum(lazy::equals(x, y));
//     EXPECT_EQ(12, (int)z);
// }
