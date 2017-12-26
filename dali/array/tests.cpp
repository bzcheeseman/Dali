#include <chrono>
#include <vector>
#include <iomanip>
#include <gtest/gtest.h>
#include "dali/config.h"

#include "dali/utils/print_utils.h"
#include "dali/runtime_config.h"
#include "dali/array/test_utils.h"
#include "dali/utils/core_utils.h"
#include "dali/array/op/dot.h"
#include "dali/array/op/binary.h"
#include "dali/array/op/reducers.h"
#include "dali/array/op/arange.h"
#include "dali/array/op/eye.h"
#include "dali/array/expression/assignment.h"
#include "dali/array/expression/buffer_view.h"

int vector_dot(Array left, Array right) {
    int out = 0;
    for (int i = 0; i < left.shape()[0]; i++) {
        out += (
            (*((int*)left[i].memory()->readonly_data(memory::Device::cpu()))) *
            (*((int*)right[i].memory()->readonly_data(memory::Device::cpu())))
        );
    }
    return out;
}

Array reference_dot(Array left, Array right) {
    Array out = Array::zeros({left.shape()[0], right.shape()[1]}, DTYPE_INT32);
    int* cpu_data_ptr = (int*)out.memory()->overwrite_data(memory::Device::cpu());
    auto strides = out.normalized_strides();
    for (int i = 0; i < left.shape()[0]; i++) {
        for (int j = 0; j < right.shape()[1]; j++) {
            cpu_data_ptr[i * strides[0] + j] = vector_dot(left[i], right[Slice()][j]);
        }
    }
    return out;
}

TEST(ArrayTests, ones) {
    memory::WithDevicePreference dp(memory::Device::cpu());
    auto x = Array::ones({3, 3}, DTYPE_INT32);
    auto y = Array({3, 3}, DTYPE_INT32);
    y = 1.0;
    EXPECT_TRUE((bool)((int)op::all_equals(x, y)));
}

TEST(ArrayTests, dot) {
    {
        memory::WithDevicePreference dp(memory::Device::cpu());
        auto x = Array::ones({3, 3}, DTYPE_INT32);
        auto y = op::dot(x, x);
        auto y_ref = reference_dot(x, x);
        auto op = op::all_equals(y, y_ref);
        EXPECT_TRUE((bool)((int)op::all_equals(y, y_ref)));
    }
#ifdef DALI_USE_CUDA
    {
        memory::WithDevicePreference dp(memory::Device::gpu(0));
        auto x = Array::ones({3, 3}, DTYPE_INT32);
        auto y = op::dot(x, x);
        auto y_ref = reference_dot(x, x);
        auto op = op::all_equals(y, y_ref);
        EXPECT_TRUE((bool)((int)op::all_equals(y, y_ref)));
    }
#endif
}

TEST(ArrayTests, autoreduce_assign) {
    Array autoreduce_assign_dest = Array::zeros(
        {3, 2, 1, 2, 1}, DTYPE_FLOAT
    );
    Array autoreduce_assign_source = Array::ones(
        {3, 2, 10, 2, 10}, DTYPE_FLOAT
    );
    Array expected_result = op::eltmul(Array::ones(
        {3, 2, 1, 2, 1}, DTYPE_FLOAT
    ), 100);

    EXPECT_TRUE((bool)((int)op::all_equals(
        op::eltmul(Array::ones({3, 2, 2}, DTYPE_FLOAT), 100),
        op::sum(autoreduce_assign_source, {2, 4})
    )));
    autoreduce_assign_dest = op::autoreduce_assign(
        autoreduce_assign_dest, autoreduce_assign_source);
    EXPECT_TRUE((bool)((int)op::all_equals(
        autoreduce_assign_dest,
        op::sum(autoreduce_assign_source, {2, 4}).expand_dims(2).expand_dims(4)
    )));
    EXPECT_TRUE((bool)((int)op::all_equals(
        expected_result,
        autoreduce_assign_dest
    )));
}

TEST(ArrayTests, scalar_value) {
    // TODO(jonathan), ensure slice parent inherits queued updates
    Array x({12}, DTYPE_INT32);
    auto assign1 = x(3) = 42;
    assign1.eval();
    EXPECT_EQ((int)x(3), 42);
    auto assign2 = x[3] = 56;
    assign2.eval();
    EXPECT_EQ((int)x(3), 56);
}

TEST(ArrayTests, slicing) {
    Array x({12});
    Array y({3,2,2});

    EXPECT_THROW(x[0][0], std::runtime_error);
    EXPECT_THROW(y[3], std::runtime_error);

    EXPECT_EQ(y[0].ndim(), 2);
    EXPECT_EQ(y[1].ndim(), 2);
    EXPECT_EQ(y[2].ndim(), 2);
    EXPECT_EQ(y[2][1].ndim(), 1);
    EXPECT_EQ(y[2][1][0].ndim(), 0);

    EXPECT_EQ(x[0].ndim(), 0);

    EXPECT_EQ(x(0).ndim(), 0);
    EXPECT_EQ(y(0).ndim(), 0);
}


TEST(ArrayTests, scalar_assign) {
    Array x = Array::zeros({3, 2}, DTYPE_INT32);
    x = 13;

    ASSERT_EQ(x.shape(), std::vector<int>({3,2}));
    ASSERT_EQ(x.dtype(), DTYPE_INT32);
    for (int i=0; i < 6; ++i) {
        if (i > 0) {
            ASSERT_TRUE(x(i).is_buffer());
        }
        ASSERT_EQ((int)x(i), 13);
    }

    x = 69.1;
    ASSERT_EQ(x.shape(), std::vector<int>({3,2}));
    ASSERT_EQ(x.dtype(), DTYPE_INT32);
    for (int i=0; i <6; ++i) {
        if (i > 0) {
            ASSERT_TRUE(x(i).is_buffer());
        }
        ASSERT_EQ((int)x(i), 69);
    }
}

TEST(ArrayTests, inplace_addition) {
    Array x = Array::zeros({3,2}, DTYPE_INT32);
    x = 13;
    x += 2;
    ASSERT_EQ((int)x.sum(), 13 * 6 + 2 * 6);
    auto prev_memory_ptr = x.memory().get();
    // add a different number in place to each element and check
    // the result is correct
    x += op::arange(6).reshape({3, 2});
    // verify that memory pointer is the same
    // (to be sure this was actually done in place)
    for (int i = 0; i < x.number_of_elements(); i++) {
        if (i > 0) {
            ASSERT_TRUE(x(i).is_buffer());
        }
        ASSERT_EQ((int)x(i), (13 + 2) + i);
    }
    ASSERT_EQ(prev_memory_ptr, x.memory().get());
}

TEST(ArrayTests, inplace_substraction) {
    Array x = Array::zeros({3,2}, DTYPE_INT32);
    x = 13;
    x -= 2;
    ASSERT_EQ((int)(Array)x.sum(), 13 * 6 - 2 * 6);

    auto prev_memory_ptr = x.memory().get();
    // add a different number in place to each element and check
    // the result is correct
    x -= op::arange(6).reshape({3, 2});
    // verify that memory pointer is the same
    // (to be sure this was actually done in place)
    for (int i = 0; i < x.number_of_elements(); i++) {
        if (i > 0) {
            ASSERT_TRUE(x(i).is_buffer());
        }
        ASSERT_EQ((int)x(i), (13 - 2) - i);
    }
    ASSERT_EQ(prev_memory_ptr, x.memory().get());
}

TEST(ArrayTests, inplace_multiplication) {
    Array x = Array::zeros({3,2}, DTYPE_INT32);
    x = 13;
    x *= 2;
    ASSERT_EQ((int)(Array)x.sum(), 13 * 6 * 2);

    auto prev_memory_ptr = x.memory().get();
    // add a different number in place to each element and check
    // the result is correct
    x *= op::arange(6).reshape({3, 2});
    // verify that memory pointer is the same
    // (to be sure this was actually done in place)
    for (int i = 0; i < x.number_of_elements(); i++) {
        if (i > 0) {
            ASSERT_TRUE(x(i).is_buffer());
        }
        ASSERT_EQ((int)x(i), (13 * 2) * i);
    }
    ASSERT_EQ(prev_memory_ptr, x.memory().get());
}


TEST(ArrayTests, scalar_construct) {
    Array scalar = float(3.14);
    ASSERT_EQ(scalar.shape(), std::vector<int>());
    ASSERT_EQ(scalar.dtype(), DTYPE_FLOAT);
    ASSERT_NEAR((float)scalar(0), 3.14, 1e-6);

    Array scalar2 = double(3.14);
    ASSERT_EQ(scalar2.shape(), std::vector<int>());
    ASSERT_EQ(scalar2.dtype(), DTYPE_DOUBLE);
    ASSERT_NEAR((double)scalar2(0), 3.14, 1e-6);

    Array scalar3 = 314;
    ASSERT_EQ(scalar3.shape(), std::vector<int>());
    ASSERT_EQ(scalar3.dtype(), DTYPE_INT32);
    ASSERT_EQ((int)scalar3(0), 314);
}


TEST(ArrayTest, eye_init_chunked) {
    Array myeye = Array({4, 5}, DTYPE_INT32);//[Slice()][Slice({}, {}, -1)];
    double diag = 5.0;

    // initialize with different diagonal values:
    myeye = op::assign(myeye, OPERATOR_T_EQL, op::diag(diag, 4, 5));
    for (int i = 0; i < myeye.shape()[0]; i++) {
        for (int j = 0; j < myeye.shape()[1]; j++) {
            auto el = myeye[i][j];
            if (i > 0 | j > 0) {
                ASSERT_TRUE(el.is_buffer());
            }
            ASSERT_EQ(el.shape().size(), 0);
            ASSERT_EQ(i == j ? diag : 0.0, (int)el);
            ASSERT_EQ(el.shape().size(), 0);
        }
    }
    myeye.eval();
    // operate on Array using identity initialization:
    myeye -= op::diag(1.0, 4, 5);

    for (int i = 0; i < myeye.shape()[0]; i++) {
        for (int j = 0; j < myeye.shape()[1]; j++) {
            auto el = myeye[i][j];
            if (i > 0 | j > 0) {
                ASSERT_TRUE(el.is_buffer());
            }
            ASSERT_EQ(el.shape().size(), 0);
            ASSERT_EQ(i == j ? (diag - 1.0) : 0.0, (int)el);
            ASSERT_EQ(el.shape().size(), 0);
        }
    }
}

TEST(ArrayTest, eye_init_composite) {
    Array myeye = Array({4, 5}, DTYPE_INT32);//[Slice()][Slice({}, {}, -1)];
    double diag = 5.0;

    // initialize with different diagonal values:
    myeye = op::assign(myeye, OPERATOR_T_EQL, op::diag(diag, 4, 5));
    for (int i = 0; i < myeye.shape()[0]; i++) {
        for (int j = 0; j < myeye.shape()[1]; j++) {
            auto el = myeye[i][j];
            if (i > 0 | j > 0) {
                ASSERT_TRUE(el.is_buffer());
            }
            ASSERT_EQ(el.shape().size(), 0);
            ASSERT_EQ(i == j ? diag : 0.0, (int)el);
            ASSERT_EQ(el.shape().size(), 0);
        }
    }
    // operate on Array using identity initialization:
    myeye -= op::diag(1.0, 4, 5);

    for (int i = 0; i < myeye.shape()[0]; i++) {
        for (int j = 0; j < myeye.shape()[1]; j++) {
            auto el = myeye[i][j];
            if (i > 0 | j > 0) {
                ASSERT_TRUE(el.is_buffer());
            }
            ASSERT_EQ(el.shape().size(), 0);
            ASSERT_EQ(i == j ? (diag - 1.0) : 0.0, (int)el);
            ASSERT_EQ(el.shape().size(), 0);
        }
    }
}

bool spans_entire_memory(Array x) {
    x.eval();
    return op::static_as_buffer_view(x)->spans_entire_memory();
}

TEST(ArrayTests, spans_entire_memory) {
    // an array is said to span its entire memory
    // if it is not a "view" onto said memory.

    // the following 3D tensor spans its entire memory
    // (in fact it even allocated it!)
    Array x = Array::zeros({3,2,2});
    ASSERT_TRUE(spans_entire_memory(x));

    // however a slice of x may not have the same property:
    auto subx = x[0];
    ASSERT_FALSE(spans_entire_memory(subx));

    // Now let's take a corner case:
    // the leading dimension of the following
    // array is 1, so taking a view at "row" 0
    // makes no difference in terms of underlying
    // memory hence, both it and its subview will
    // "span the entire memory"
    Array y = Array::zeros({1,2,2});
    ASSERT_TRUE(spans_entire_memory(y));

    auto view_onto_y = y[0];
    ASSERT_TRUE(spans_entire_memory(view_onto_y));

    // extreme corner case, reversed:
    Array z = Array::zeros({4});

    Array z_reversed = z[Slice({}, {}, -1)];
    ASSERT_TRUE(spans_entire_memory(z_reversed));

    // same underlying storage:
    ASSERT_EQ(z_reversed.memory().get(), z.memory().get());

    // another edge case:
    Array z2 = Array::zeros({1, 4, 1});

    Array z2_reversed = z2[Slice({}, {}, -1)][Slice({}, {}, -1)][Slice({}, {}, 2)];
    ASSERT_TRUE(spans_entire_memory(z2_reversed));
    Array z2_reversed_skip = z2[Slice({}, {}, -1)][Slice({}, {}, -2)][Slice({}, {}, 2)];
    ASSERT_FALSE(spans_entire_memory(z2_reversed_skip));
}

void inplace_add_ones(Array array) {
    array += 1;
}

TEST(ArrayTests, persist_inplace_operations) {
    Array original = Array::zeros({5, 5}, DTYPE_INT32);
    inplace_add_ones(original);
    ASSERT_TRUE((bool)(int)op::all_equals(original, Array::ones_like(original)));
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
    return op::arange(24).reshape({2, 3, 4});
}

TEST(ArrayTests, copy_constructor_force_copy) {
    Array original = op::arange(9).reshape({3, 3});
    Array copy(original, true);
    copy += 1;
    for (int i = 0; i < original.number_of_elements(); i++) {
        if (i > 0) {
            ASSERT_TRUE(original(i).is_buffer());
            ASSERT_TRUE(copy(i).is_buffer());
        }
        // since +1 was done after copy
        // change is not reflected
        EXPECT_NE((int)original(i), (int)copy(i));
    }
}

TEST(ArrayTests, copy_constructor_force_reference) {
    Array original = op::arange(9).reshape({3, 3});
    Array copy(original, false);
    copy += 1;
    for (int i = 0;  i < original.number_of_elements(); i++) {
        if (i > 0) {
            ASSERT_TRUE(original(i).is_buffer());
            ASSERT_TRUE(copy(i).is_buffer());
        }
        // copy is a view, so +1 affects both
        EXPECT_EQ((int)original(i), (int)copy(i));
    }
}

TEST(ArrayTests, copy_constructor) {
    Array original = Array({3}, DTYPE_INT32)[Slice()][Broadcast()];
    // perform copy of broadcasted data
    Array hard_copy(original, true);
    EXPECT_EQ(original.shape(), hard_copy.shape());

    Array soft_copy(original, false);
    EXPECT_EQ(original.shape(), soft_copy.shape());

    // 'operator=' uses soft copy too:
    auto soft_copy_assign = original;
    EXPECT_EQ(original.shape(), soft_copy_assign.shape());
}

#ifdef DONT_COMPILE


TEST(ArrayTests, reshape_with_unknown_dimension) {
    Array x({2, 3, 4}, DTYPE_INT32);
    auto right_deduce = x.reshape({6,-1});
    EXPECT_EQ(std::vector<int>({6, 4}), right_deduce.shape());
    auto left_deduce = x.reshape({-1,2});
    EXPECT_EQ(std::vector<int>({12, 2}), left_deduce.shape());
    // too many unknowns:
    EXPECT_THROW(x.reshape({-1,-1}), std::runtime_error);
    // shape is incompatible
    EXPECT_THROW(x.reshape({-1,5}), std::runtime_error);
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
    ASSERT_THROW(Slice(0,{},1).size(),  std::runtime_error);
}

TEST(ArrayTests, slice_contains) {
    EXPECT_TRUE(Slice(0,12,2).contains(0));
    EXPECT_FALSE(Slice(0,12,2).contains(1));

    EXPECT_FALSE(Slice(0,12,-2).contains(0));
    EXPECT_TRUE(Slice(0,12,-2).contains(1));

    ASSERT_THROW(Slice(0, {}).contains(1),  std::runtime_error);
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

TEST(ArrayTests, inplace_strided_addition) {
    auto x = build_234_arange();
    auto x_plucked = x.pluck_axis(2, 1);
    // strided dimension pluck is a view
    EXPECT_EQ(x_plucked.memory().get(), x.memory().get());
    // we now modify this view by in-place incrementation:
    // sum was originally 66, should now be 72:
    x_plucked += 1;
    // sum is now same as before + number of elements
    EXPECT_EQ(
        (int)(Array)x_plucked.sum(),
        x_plucked.number_of_elements() + (1 + 5 + 9 + 13 + 17 + 21)
    );
}

TEST(ArrayTests, canonical_reshape) {
    ASSERT_EQ(mshadow::Shape1(60),        internal::canonical_reshape<1>({3,4,5}));
    ASSERT_EQ(mshadow::Shape2(12,5),      internal::canonical_reshape<2>({3,4,5}));
    ASSERT_EQ(mshadow::Shape3(3,4,5),     internal::canonical_reshape<3>({3,4,5}));
    ASSERT_EQ(mshadow::Shape4(1,3,4,5),   internal::canonical_reshape<4>({3,4,5}));
}

std::vector<Slice> generate_interesting_slices(int dim_size) {
    std::vector<Slice> interesting_slices;
    for (int start = 0; start < dim_size; ++start) {
        for (int end = start + 1; end <= dim_size; ++end) {
            for (int step = -2; step < 3; ++step) {
                if (step == 0) continue;
                interesting_slices.push_back(Slice(start,end,step));
            }
        }
    }
    EXPECT_TRUE(interesting_slices.size() < 50);
    return interesting_slices;
}

TEST(ArrayTests, proper_slicing) {
    Array x = build_234_arange();
    Array sliced = x[Slice(0,-1)][2][Slice(0,4,-2)];

    Array sliced_sum = sliced.sum();
    ASSERT_EQ(20, (int)sliced_sum);
}

TEST(ArrayTests, double_striding) {
    const int NRETRIES = 2;
    for (int retry=0; retry < NRETRIES; ++retry) {

        Array x({2,3,4}, DTYPE_INT32);
        x = initializer::uniform(-1000, 1000);

        for (auto& slice0: generate_interesting_slices(2)) {
            for (auto& slice1: generate_interesting_slices(3)) {
                for (auto& slice2: generate_interesting_slices(4)) {
                    // SCOPED_TRACE(std::string(utils::MS() << "x[" << slice0 << "][" << slice1 <<  "][" <<  slice2 << "]"));
                    Array sliced = x[slice0][slice1][slice2];
                    int actual_sum = (Array)sliced.sum();
                    int expected_sum = 0;
                    for (int i=0; i < 2; ++i) {
                        for (int j=0; j<3; ++j) {
                            for (int k=0; k<4; ++k) {
                                if (slice0.contains(i) && slice1.contains(j) && slice2.contains(k)) {
                                    // avoiding the use of [] here because [] itself
                                    // does striding.
                                    expected_sum += (int)x(i*12 + j*4 + k);
                                }
                            }
                        }
                    }
                    ASSERT_EQ(expected_sum, actual_sum);
                }
            }
        }
    }
}

TEST(ArrayLazyOpsTests, reshape_broadcasted) {
    auto B = Array::ones({3},     DTYPE_INT32);

    B = B[Broadcast()][Slice()][Broadcast()];
    B = B.reshape_broadcasted({2,3,4});

    ASSERT_EQ((int)(Array)B.sum(), 2 * 3 * 4);
}

TEST(ArrayLazyOpsTests, reshape_broadcasted2) {
    auto B = Array::ones({3},     DTYPE_INT32);
    B = B[Broadcast()][Slice()][Broadcast()];

    B = B.reshape_broadcasted({2, 3, 1});
    B = B.reshape_broadcasted({2, 3, 1});
    B = B.reshape_broadcasted({2, 3, 5});
    B = B.reshape_broadcasted({2, 3, 5});

    EXPECT_THROW(B.reshape_broadcasted({5,3,5}), std::runtime_error);
    EXPECT_THROW(B.reshape_broadcasted({1,3,5}), std::runtime_error);
    EXPECT_THROW(B.reshape_broadcasted({2,3,1}), std::runtime_error);
}

TEST(ArrayTests, strides_compacted_after_expansion) {
    Array x = Array::zeros({2,3,4});

    EXPECT_EQ(x.expand_dims(0).strides(), vector<int>());
    EXPECT_EQ(x.expand_dims(1).strides(), vector<int>());
    EXPECT_EQ(x.expand_dims(2).strides(), vector<int>());
    EXPECT_EQ(x.expand_dims(3).strides(), vector<int>());
}

// use bracket to compute flat sequence of element in array.
void sequence_array(Array x, std::vector<int>& output) {
    if (x.ndim() == 0) {
        output.push_back((int)x);
    } else {
        for (int i = 0; i < x.shape()[0]; ++i) {
            sequence_array(x[i], output);
        }
    }
}

void ensure_call_operator_correct(Array x) {
    std::vector<int> correct_elem_sequence;

    sequence_array(x, correct_elem_sequence);
    for (int i = 0; i < x.number_of_elements(); ++i) {
        EXPECT_EQ(correct_elem_sequence[i], (int)x(i));
    }
}

TEST(ArrayTest, strided_call_operator) {
    Array x = build_234_arange();
    ensure_call_operator_correct(x);

    Array x2 = x[Slice()][2];
    ensure_call_operator_correct(x2);

    Array x3 = x[Slice({},{}, -1)][2];
    ensure_call_operator_correct(x2);

    Array y({2, 2}, DTYPE_INT32);
    y = initializer::arange(0, 1);
    ensure_call_operator_correct(y);

    Array y2 = y[Slice()][Broadcast()][Slice()];
    ensure_call_operator_correct(y2);

    Array y3 = y2.reshape_broadcasted({2,3,2});
    ensure_call_operator_correct(y3);
}

TEST(ArrayTest, transpose) {
    Array x = Array::zeros({2},     DTYPE_INT32);
    Array y = Array::zeros({2,3},   DTYPE_INT32);
    Array z = Array::zeros({2,3,4}, DTYPE_INT32);

    auto x_T = x.transpose();
    auto y_T = y.transpose();
    auto z_T = z.transpose();

    ASSERT_EQ(VI({2}),     x_T.shape());
    ASSERT_EQ(VI({3,2}),   y_T.shape());
    ASSERT_EQ(VI({4,3,2}), z_T.shape());

    for (int i = 0; i < 2; ++i) {
        ASSERT_EQ((int)x[i], (int)x_T[i]);
    }

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            ASSERT_EQ((int)y[i][j], (int)y_T[j][i]);
        }
    }

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 4; ++k) {
                ASSERT_EQ((int)z[i][j][k], (int)z_T[k][j][i]);
            }
        }
    }

    auto z_T_funny = z.transpose({1,0,2});
    ASSERT_EQ(VI({3,2,4}), z_T_funny.shape());

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 4; ++k) {
                ASSERT_EQ((int)z[i][j][k], (int)z_T_funny[j][i][k]);
            }
        }
    }
}

TEST(ArrayTests, is_transpose) {
    Array a0 = Array::zeros({},     DTYPE_INT32);
    Array a1 = Array::zeros({2},     DTYPE_INT32);
    Array a2 = Array::zeros({2,3},     DTYPE_INT32);
    Array a3 = Array::zeros({2,3,4},     DTYPE_INT32);
    Array a4 = Array::zeros({2,3,4,5},     DTYPE_INT32);
    ASSERT_TRUE(a0.transpose().is_transpose());
    ASSERT_TRUE(a1.transpose().is_transpose());
    ASSERT_TRUE(a2.transpose().is_transpose());
    ASSERT_TRUE(a3.transpose().is_transpose());
    ASSERT_TRUE(a4.transpose().is_transpose());

    ASSERT_FALSE(a2.is_transpose());
    ASSERT_FALSE(a3.is_transpose());
    ASSERT_FALSE(a4.is_transpose());

    Array a4_with_jumps = a4[Slice()][Slice()][Slice({},{},2)];
    ASSERT_FALSE(a4_with_jumps.is_transpose());
    ASSERT_FALSE(a4_with_jumps.transpose().is_transpose());
}

TEST(ArrayIOTests, detect_types_on_load) {
    auto loaded_ints = Array::load(
        utils::dir_join({STR(DALI_DATA_DIR), "tests", "vals_int.npy"})
    );
    EXPECT_EQ(DTYPE_INT32, loaded_ints.dtype());
    auto loaded_floats = Array::load(
        utils::dir_join({STR(DALI_DATA_DIR), "tests", "vals_float.npy"})
    );
    EXPECT_EQ(DTYPE_FLOAT, loaded_floats.dtype());
    auto loaded_doubles = Array::load(
        utils::dir_join({STR(DALI_DATA_DIR), "tests", "vals_double.npy"})
    );
    EXPECT_EQ(DTYPE_DOUBLE, loaded_doubles.dtype());
}

TEST(ArrayIOTests, load_fortran) {
    auto arange = Array::load(
        utils::dir_join({STR(DALI_DATA_DIR), "tests", "arange12.npy"})
    );
    auto arange_fortran = Array::load(
        utils::dir_join({STR(DALI_DATA_DIR), "tests", "arange12.fortran.npy"})
    );

    ASSERT_TRUE(Array::equals(arange, arange_fortran));
    for (int i = 0; i < 12; i++) {
        EXPECT_EQ_DTYPE(i, arange_fortran(i), arange_fortran.dtype());
    }
}

TEST(ArrayIOTests, save_load_test) {
    // load arange, then save it to a new file
    auto arange = Array::load(
        utils::dir_join({STR(DALI_DATA_DIR), "tests", "arange12.npy"})
    );
    Array::save(
        utils::dir_join({STR(DALI_DATA_DIR),  "tests", "arange12.temp.npy"}),
        arange
    );
    auto reloaded = Array::load(
        utils::dir_join({STR(DALI_DATA_DIR),  "tests", "arange12.temp.npy"})
    );
    ASSERT_TRUE(Array::equals(arange, reloaded));
    for (int i = 0; i < 12; i++) {
        EXPECT_EQ_DTYPE(i, arange(i), arange.dtype());
    }
}

TEST(ArrayConstructorTests, shape_preservation) {
    auto constructors = {
        Array::zeros_like,
        Array::empty_like,
        Array::ones_like
    };
    for (auto constructor : constructors) {
        Array x({3, 2, 5});
        auto y = constructor(x);

        EXPECT_EQ(x.dtype(), y.dtype());
        EXPECT_EQ(x.shape(), y.shape());
        EXPECT_EQ(x.bshape(), y.bshape());

        Array x2 = x.insert_broadcast_axis(0).insert_broadcast_axis(3);
        auto y2 = constructor(x2);

        EXPECT_EQ(x2.dtype(), y2.dtype());
        EXPECT_EQ(x2.shape(), y2.shape());
        EXPECT_EQ(x2.bshape(), y2.bshape());
    }
}

#endif
