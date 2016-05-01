#include <chrono>
#include <vector>
#include <iomanip>
#include <gtest/gtest.h>
#include "dali/config.h"
#include <mshadow/tensor.h>

#include "dali/array/function/typed_array.h"
#include "dali/array/op/unary.h"
#include "dali/array/op/binary.h"
#include "dali/array/op/other.h"
#include "dali/array/op/initializer.h"
#include "dali/utils/print_utils.h"
#include "dali/runtime_config.h"
#include "dali/array/op.h"

#include "dali/array/lazy/binary.h"

using std::vector;

TEST(ArrayTests, slicing) {
    Array x({12});
    Array y({3,2,2});

    EXPECT_THROW(x[0][0], std::runtime_error);
    EXPECT_THROW(y[3], std::runtime_error);

    EXPECT_EQ(y[0].shape().size(), 2);
    EXPECT_EQ(y[1].shape().size(), 2);
    EXPECT_EQ(y[2].shape().size(), 2);
    EXPECT_EQ(y[2][1].shape().size(), 1);
    EXPECT_EQ(y[2][1][0].shape().size(), 0);

    EXPECT_EQ(x[0].shape().size(), 0);

    EXPECT_EQ(x(0).shape().size(), 0);
    EXPECT_EQ(y(0).shape().size(), 0);
}

TEST(ArrayTests, scalar_value) {
    Array x({12}, DTYPE_INT32);
    x(3) = 42;
    int& x_val = x(3);
    EXPECT_EQ(x_val, 42);
    x[3] = 56;
    EXPECT_EQ(x_val, 56);
}

TEST(ArrayTests, scalar_assign) {
    Array x = Array::zeros({3,2}, DTYPE_INT32);
    x = 13;

    ASSERT_EQ(x.shape(), std::vector<int>({3,2}));
    ASSERT_EQ(x.dtype(), DTYPE_INT32);
    for (int i=0; i < 6; ++i) {
        ASSERT_EQ((int)x(i), 13);
    }

    x = 69.1;
    ASSERT_EQ(x.shape(), std::vector<int>({3,2}));
    ASSERT_EQ(x.dtype(), DTYPE_INT32);
    for (int i=0; i <6; ++i) {
        ASSERT_EQ((int)x(i), 69);
    }
}

TEST(ArrayTests, inplace_addition) {
    Array x = Array::zeros({3,2}, DTYPE_INT32);
    x = 13;
    x += 2;
    ASSERT_EQ((int)(Array)x.sum(), 13 * 6 + 2 * 6);

    auto prev_memory_ptr = &(*x.memory());
    // add a different number in place to each element and check
    // the result is correct
    x += Array::arange({3, 2}, DTYPE_INT32);
    // verify that memory pointer is the same
    // (to be sure this was actually done in place)
    ASSERT_EQ(prev_memory_ptr, &(*x.memory()));
    for (int i = 0; i < x.number_of_elements(); i++) {
        ASSERT_EQ((int)x(i), (13 + 2) + i);
    }
}

TEST(ArrayTests, inplace_substraction) {
    Array x = Array::zeros({3,2}, DTYPE_INT32);
    x = 13;
    x -= 2;
    ASSERT_EQ((int)(Array)x.sum(), 13 * 6 - 2 * 6);

    auto prev_memory_ptr = &(*x.memory());
    // add a different number in place to each element and check
    // the result is correct
    x -= Array::arange({3, 2}, DTYPE_INT32);
    // verify that memory pointer is the same
    // (to be sure this was actually done in place)
    ASSERT_EQ(prev_memory_ptr, &(*x.memory()));
    for (int i = 0; i < x.number_of_elements(); i++) {
        ASSERT_EQ((int)x(i), (13 - 2) - i);
    }
}

TEST(ArrayTests, inplace_multiplication) {
    Array x = Array::zeros({3,2}, DTYPE_INT32);
    x = 13;
    x *= 2;
    ASSERT_EQ((int)(Array)x.sum(), 13 * 6 * 2);

    auto prev_memory_ptr = &(*x.memory());
    // add a different number in place to each element and check
    // the result is correct
    x *= Array::arange({3, 2}, DTYPE_INT32);
    // verify that memory pointer is the same
    // (to be sure this was actually done in place)
    ASSERT_EQ(prev_memory_ptr, &(*x.memory()));
    for (int i = 0; i < x.number_of_elements(); i++) {
        ASSERT_EQ((int)x(i), (13 * 2) * i);
    }
}

TEST(ArrayTests, scalar_construct) {
    auto assignable = initializer::fill((float)3.14);
    Array scalar = assignable;
    ASSERT_EQ(scalar.shape(), std::vector<int>());
    ASSERT_EQ(scalar.dtype(), DTYPE_FLOAT);
    ASSERT_NEAR((float)scalar(0), 3.14, 1e-6);

    Array scalar2;
    scalar2 = initializer::fill((double)3.14);
    ASSERT_EQ(scalar2.shape(), std::vector<int>());
    ASSERT_EQ(scalar2.dtype(), DTYPE_DOUBLE);
    ASSERT_NEAR((double)scalar2(0), 3.14, 1e-6);

    Array scalar3 = initializer::fill(314);
    ASSERT_EQ(scalar3.shape(), std::vector<int>());
    ASSERT_EQ(scalar3.dtype(), DTYPE_INT32);
    ASSERT_EQ((int)scalar3(0), 314);
}


TEST(ArrayTests, spans_entire_memory) {
    // an array is said to span its entire memory
    // if it is not a "view" onto said memory.

    // the following 3D tensor spans its entire memory
    // (in fact it even allocated it!)
    Array x = Array::zeros({3,2,2});
    ASSERT_TRUE(x.spans_entire_memory());

    // however a slice of x may not have the same property:
    auto subx = x[0];
    ASSERT_FALSE(subx.spans_entire_memory());

    // Now let's take a corner case:
    // the leading dimension of the following
    // array is 1, so taking a view at "row" 0
    // makes no difference in terms of underlying
    // memory hence, both it and its subview will
    // "span the entire memory"
    Array y = Array::zeros({1,2,2});
    ASSERT_TRUE(y.spans_entire_memory());

    auto view_onto_y = y[0];
    ASSERT_TRUE(view_onto_y.spans_entire_memory());
}

// Some example integer 3D tensor with
// values from 0 to 23
Array build_234_arange() {
    // [
    //   [
    //     [ 0  1  2  3 ],
    //     [ 4  5  6  7 ],
    //     [ 8  9  10 11],
    //   ],
    //   [
    //     [ 12 13 14 15],
    //     [ 16 17 18 19],
    //     [ 20 21 22 23],
    //   ]
    // ]
    Array x({2,3,4}, DTYPE_INT32);
    x = initializer::arange();
    return x;
}

TEST(ArrayTests, contiguous_memory) {
    auto x = build_234_arange();
    EXPECT_TRUE(x.contiguous_memory());
}

TEST(ArrayTests, pluck_axis_stride_shape) {
    auto x = build_234_arange();

    auto x_plucked = x.pluck_axis(0, 1);
    EXPECT_EQ(x_plucked.shape(),   vector<int>({3, 4}));
    EXPECT_EQ(x_plucked.number_of_elements(), 12);
    EXPECT_EQ(x_plucked.offset(),  12    );
    // if all strides are 1, then return empty vector
    EXPECT_EQ(x_plucked.strides(), vector<int>({}));

    auto x_plucked2 = x.pluck_axis(1, 2);
    EXPECT_EQ(x_plucked2.shape(),   vector<int>({2, 4}));
    EXPECT_EQ(x_plucked2.number_of_elements(), 8);
    EXPECT_EQ(x_plucked2.offset(),   8    );
    EXPECT_EQ(x_plucked2.strides(), vector<int>({12, 1}));

    auto x_plucked3 = x.pluck_axis(2, 1);
    EXPECT_EQ(x_plucked3.shape(),   vector<int>({2, 3}));
    EXPECT_EQ(x_plucked3.number_of_elements(), 6);
    EXPECT_EQ(x_plucked3.offset(),  1);
    EXPECT_EQ(x_plucked3.strides(), vector<int>({12, 4}));
}


TEST(ArrayTests, slice_size) {
    ASSERT_EQ(5, Slice(0,5).size());
    ASSERT_EQ(2, Slice(2,4).size());
    ASSERT_EQ(3, Slice(0,5,2).size());
    ASSERT_EQ(3, Slice(0,5,-2).size());
    ASSERT_EQ(2, Slice(0,6,3).size());
    ASSERT_EQ(2, Slice(0,6,-3).size());
    ASSERT_EQ(3, Slice(0,7,3).size());
    ASSERT_EQ(3, Slice(0,7,-3).size());

    ASSERT_THROW(Slice(0,2,0),  std::runtime_error);
}

TEST(ArrayTests, pluck_axis_eval) {
    auto x = build_234_arange();

    auto x_plucked = x.pluck_axis(0, 0);
    EXPECT_EQ(x.memory().get(), x_plucked.memory().get());
    EXPECT_EQ(
        (int)(Array)x_plucked.sum(),
        0 + 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10 + 11
    );

    auto x_plucked2 = x.pluck_axis(1, 2);
    EXPECT_EQ(x.memory().get(), x_plucked2.memory().get());
    EXPECT_FALSE(x_plucked2.contiguous_memory());
    EXPECT_EQ(
        (int)(Array)x_plucked2.sum(),
        8 + 9 + 10 + 11 + 20 + 21 + 22 + 23
    );

    auto x_plucked3 = x.pluck_axis(2, 1);

    EXPECT_EQ(x.memory().get(), x_plucked3.memory().get());
    EXPECT_FALSE(x_plucked3.contiguous_memory());
    EXPECT_EQ(
        (int)(Array)x_plucked3.sum(),
        1 + 5 + 9 + 13 + 17 + 21
    );
}

TEST(ArrayTests, DISABLED_inplace_strided_addition) {
    auto x = build_234_arange();
    auto x_plucked = x.pluck_axis(2, 1);
    // strided dimension pluck is a view
    EXPECT_EQ(&(*x_plucked.memory()), &(*x.memory()));
    // we now modify this view by in-place incrementation:
    // sum was originally 66, should now be 72:
    x_plucked += 1;
    // sum is now same as before + number of elements
    EXPECT_EQ(
        (int)(Array)x_plucked.sum(),
        (1 + 5 + 9 + 13 + 17 + 21) + x_plucked.number_of_elements()
    );
}

TEST(ArrayTests, canonical_reshape) {
    ASSERT_EQ(mshadow::Shape1(60),        internal::canonical_reshape<1>({3,4,5}));
    ASSERT_EQ(mshadow::Shape2(12,5),      internal::canonical_reshape<2>({3,4,5}));
    ASSERT_EQ(mshadow::Shape3(3,4,5),     internal::canonical_reshape<3>({3,4,5}));
    ASSERT_EQ(mshadow::Shape4(1,3,4,5),   internal::canonical_reshape<4>({3,4,5}));
}
