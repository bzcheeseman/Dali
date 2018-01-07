#include <chrono>
#include <vector>
#include <iomanip>
#include <gtest/gtest.h>
#include "dali/config.h"

#include "dali/utils/print_utils.h"
#include "dali/runtime_config.h"
#include "dali/array/test_utils.h"
#include "dali/utils/core_utils.h"
#include "dali/utils/make_message.h"
#include "dali/array/op/dot.h"
#include "dali/array/op/binary.h"
#include "dali/array/op/unary.h"
#include "dali/array/op/reducers.h"
#include "dali/array/op/arange.h"
#include "dali/array/op/eye.h"
#include "dali/array/op/uniform.h"
#include "dali/array/op/outer.h"
#include "dali/array/op/one_hot.h"
#include "dali/array/op/gather.h"
#include "dali/array/op/gather_from_rows.h"
#include "dali/array/jit/scalar_view.h"
#include "dali/array/expression/assignment.h"
#include "dali/array/expression/buffer_view.h"
#include "dali/array/expression/control_flow.h"
#include "dali/array/functor.h"

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
        Array::ones({3, 2, 2}, DTYPE_FLOAT) * 100,
        op::sum(autoreduce_assign_source, {2, 4})
    )));
    // autoreduce_assign_dest = op::autoreduce_assign(
    //     autoreduce_assign_dest, autoreduce_assign_source);
    // EXPECT_TRUE((bool)((int)op::all_equals(
    //     autoreduce_assign_dest,
    //     op::sum(autoreduce_assign_source, {2, 4}).expand_dims(2).expand_dims(4)
    // )));
    // EXPECT_TRUE((bool)((int)op::all_equals(
    //     expected_result,
    //     autoreduce_assign_dest
    // )));
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
    copy += 2;
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
    EXPECT_EQ(x_plucked.shape(),   std::vector<int>({3, 4}));
    EXPECT_EQ(x_plucked.number_of_elements(), 12);
    EXPECT_EQ(x_plucked.offset(),  12    );
    // if all strides are 1, then return empty vector
    EXPECT_EQ(x_plucked.strides(), std::vector<int>({}));

    auto x_plucked2 = x.pluck_axis(1, 2);
    EXPECT_EQ(x_plucked2.shape(),   std::vector<int>({2, 4}));
    EXPECT_EQ(x_plucked2.number_of_elements(), 8);
    EXPECT_EQ(x_plucked2.offset(),   8    );
    EXPECT_EQ(x_plucked2.strides(), std::vector<int>({12, 1}));

    auto x_plucked3 = x.pluck_axis(2, 1);
    EXPECT_EQ(x_plucked3.shape(),   std::vector<int>({2, 3}));
    EXPECT_EQ(x_plucked3.number_of_elements(), 6);
    EXPECT_EQ(x_plucked3.offset(),  1);
    EXPECT_EQ(x_plucked3.strides(), std::vector<int>({12, 4}));
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

// TODO(jonathan): ensure this test also includes
// unremovable assignments (e.g. inplace operation done below)
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
    Array sliced = x[Slice(0,-1)][2][Slice(0, 4, -2)];
    Array sliced_sum = sliced.sum();
    ASSERT_EQ(20, (int)sliced_sum);
}

TEST(ArrayTests, double_striding) {
    const int NRETRIES = 2;
    for (int retry=0; retry < NRETRIES; ++retry) {

        Array x = op::uniform(-1000, 1000, {2, 3, 4});

        for (auto& slice0: generate_interesting_slices(2)) {
            for (auto& slice1: generate_interesting_slices(3)) {
                for (auto& slice2: generate_interesting_slices(4)) {
                    SCOPED_TRACE(utils::make_message("x[", slice0, "][", slice1, "][", slice2, "]"));
                    Array sliced = x[slice0][slice1][slice2];
                    int actual_sum = sliced.sum();
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

TEST(ArrayLazyOpsTests, sum_broadcasting) {
    auto B = Array::ones({3}, DTYPE_INT32);

    B = B[Broadcast()][Slice()][Broadcast()];
    B = op::add(B, Array::zeros({2,3,4}));

    ASSERT_EQ((int)(Array)B.sum(), 2 * 3 * 4);
}

TEST(ArrayLazyOpsTests, broadcast_to_shape) {
    auto B = Array::ones({3}, DTYPE_INT32);
    B = B[Broadcast()][Slice()][Broadcast()];

    B = B.broadcast_to_shape({2, 3, 1});
    B = B.broadcast_to_shape({2, 3, 1});
    B = B.broadcast_to_shape({2, 3, 5});
    B = B.broadcast_to_shape({2, 3, 5});

    EXPECT_THROW(B.broadcast_to_shape({5,4,5}), std::runtime_error);
}


TEST(ArrayTests, strides_compacted_after_expansion) {
    Array x = Array::zeros({2,3,4});

    EXPECT_EQ(x.expand_dims(0).strides(), std::vector<int>());
    EXPECT_EQ(x.expand_dims(1).strides(), std::vector<int>());
    EXPECT_EQ(x.expand_dims(2).strides(), std::vector<int>());
    EXPECT_EQ(x.expand_dims(3).strides(), std::vector<int>());
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

    Array y = op::arange(4).reshape({2, 2});
    ensure_call_operator_correct(y);

    Array y2 = y[Slice()][Broadcast()][Slice()];
    ensure_call_operator_correct(y2);

    Array y3 = y2.broadcast_to_shape({2,3,2});
    ensure_call_operator_correct(y3);
}


TEST(ArrayTest, transpose) {
    Array x = Array::zeros({2},     DTYPE_INT32);
    Array y = Array::zeros({2,3},   DTYPE_INT32);
    Array z = Array::zeros({2,3,4}, DTYPE_INT32);

    auto x_T = x.transpose();
    auto y_T = y.transpose();
    auto z_T = z.transpose();

    ASSERT_EQ(std::vector<int>({2}),     x_T.shape());
    ASSERT_EQ(std::vector<int>({3,2}),   y_T.shape());
    ASSERT_EQ(std::vector<int>({4,3,2}), z_T.shape());

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
    ASSERT_EQ(std::vector<int>({3,2,4}), z_T_funny.shape());

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

        Array x2 = x.insert_broadcast_axis(0).insert_broadcast_axis(3);
        auto y2 = constructor(x2);

        EXPECT_EQ(x2.dtype(), y2.dtype());
        EXPECT_EQ(x2.shape(), y2.shape());
    }
}


TEST(ArrayTests, lse_and_addition) {
    Array s1 = Array::ones({3,4}, DTYPE_INT32);
    Array s2 = Array::ones({3,4}, DTYPE_INT32);
    Array target = Array::zeros({3,4}, DTYPE_INT32);
    target <<= s1 + s2;

    for (int i = 0; i < target.number_of_elements(); ++i) {
        EXPECT_EQ(2, (int)target(i));
    }
}

TEST(ArrayTests, lse_3D) {
    Array target = Array({3,2,2});
    Array source = Array({3,2,2});
    target <<= source;
}

TEST(ArrayTests, lse) {
    Array target = Array::zeros({3,4}, DTYPE_INT32);
    Array source = op::arange(12).reshape({3,4});

    target <<= source;

    for (int i = 0; i < target.number_of_elements(); ++i) {
        EXPECT_EQ(i, (int)target(i));
    }
}

TEST(ArrayTests, broadcasted_lse) {
    Array target = Array::zeros({3}, DTYPE_INT32)[Slice(0,3)][Broadcast()];
    Array source = Array::ones({3,4}, DTYPE_INT32);

    target <<= source;

    for (int i = 0; i < target.number_of_elements(); ++i) {
        EXPECT_EQ(4,  (int)target(i));
    }
}

TEST(ArrayTests, broadcasted_lse2) {
    Array target = Array::zeros({4}, DTYPE_INT32)[Broadcast()][Slice(0,4)];
    Array source = Array::ones({3,4}, DTYPE_INT32);

    target <<= source;

    for (int i = 0; i < target.number_of_elements(); ++i) {
        EXPECT_EQ(3,  (int)target(i));
    }
}


TEST(ArrayCastTests, astype) {
    Array integer_arange = op::arange(6).reshape({1,2,3});

    EXPECT_EQ(DTYPE_INT32, integer_arange.dtype());

    Array float_arange_with_offset = (Array)integer_arange.astype(DTYPE_FLOAT) - 0.6;

    EXPECT_EQ(DTYPE_FLOAT, float_arange_with_offset.dtype());

    for (int i = 0; i < integer_arange.number_of_elements(); i++) {
        EXPECT_NEAR((float)integer_arange(i) - 0.6, (float)float_arange_with_offset(i), 1e-6);
    }

    Array int_arange_with_offset = float_arange_with_offset.astype(DTYPE_INT32);

    for (int i = 0; i < integer_arange.number_of_elements(); i++) {
        EXPECT_EQ(std::round((float)integer_arange(i) - 0.6), (int)int_arange_with_offset(i));
    }
}

TEST(ArrayCastTests, mean) {
    Array integer_arange = op::arange(6).reshape({1,2,3});
    Array mean = integer_arange.mean();
    EXPECT_EQ(DTYPE_DOUBLE, mean.dtype());
    EXPECT_EQ((0.0 + 1.0 + 2.0 + 3.0 + 4.0 + 5.0)/6.0, (double)mean(0));

    Array mean_axis = integer_arange.mean(-1);

    EXPECT_EQ(DTYPE_DOUBLE, mean_axis.dtype());
    EXPECT_EQ((0.0 + 1.0 + 2.0)/3.0, (double)mean_axis(0));
    EXPECT_EQ((3.0 + 4.0 + 5.0)/3.0, (double)mean_axis(1));
}


TEST(ArrayBinaryTests, lazy_binary_correctness) {
    Array x({2,1});
    Array y({2,1});
    Array z({2,1});

    std::vector<Array> assigns = {
        x(0) = 2,
        y(0) = 3,
        z(0) = 5,
        x(1) = 7,
        y(1) = 11,
        z(1) = 13
    };
    for (auto assign : assigns) {assign.eval();}

    auto partial = x * y * z;
    Array res = partial;

    EXPECT_EQ((float)(res(0)), 2 * 3  * 5);
    EXPECT_EQ((float)(res(1)), 7 * 11 * 13);
}


TEST(ArrayBinaryTests, broadcasted_add) {
    auto out = Array::zeros({2,3,4}, DTYPE_INT32);
    auto A = Array::ones({2,3,4}, DTYPE_INT32);
    auto B = Array::ones({3},     DTYPE_INT32);

    B = B[Broadcast()][Slice(0,3)][Broadcast()];

    out = A + 2 * B;

    ASSERT_EQ((int)(Array)out.sum(), 2 * 3 * 4 * 3);
}

TEST(ArrayBinaryTests, advanced_striding_with_reductions) {
    Array x = op::arange(12).reshape({3,4});
    Array y = op::arange(12).reshape({3,4});
    y = y[Slice(0,3)][Slice(0,4,-1)];
    for (int i =0; i < 12; ++i) (y(i) = i).eval();

    Array z =  op::sum(op::equals(x, y));
    EXPECT_EQ(12, (int)z);
}

TEST(ArrayBinaryTests, advanced_striding_with_reductions1) {
    Array x = op::arange(12).reshape({3,4});
    Array y = op::arange(12).reshape({3,4});
    y = y[Slice(0,3,-1)];
    for (int i =0; i <12; ++i) (y(i) = i).eval();

    Array z = op::sum(op::equals(x, y));
    EXPECT_EQ(12, (int)z);
}

TEST(ArrayBinaryTests, advanced_striding_with_reductions2) {
    Array x = op::arange(12);
    Array y_source = op::arange(24).reshape({12,2});
    Array y = y_source[Slice(0,12)][1];

    for (int i =0; i <12; ++i) (y(i) = i).eval();

    Array z =  op::sum(op::equals(x, y));
    EXPECT_EQ(12, (int)z);
}


namespace {
    Array reference_outer_product(const Array& left, const Array& right) {
        ASSERT2(left.ndim() == 1 && right.ndim() == 1, "left and right should have ndim == 1");
        Array out = Array::zeros({left.shape()[0], right.shape()[0]}, left.dtype());
        for (int i = 0; i < out.shape()[0]; i++) {
            for (int j = 0; j < out.shape()[1]; j++) {
                op::assign(out[i][j], OPERATOR_T_EQL, left[i] * right[j]).eval();
            }
        }
        return out;
    }
}

TEST(ArrayTests, outer_product_chainable) {
    Array x = op::arange(3);
    Array y = op::arange(4);
    Array outer = op::outer(op::tanh(x - 3.0), op::tanh(y - 2.0));
    auto expected_outer = reference_outer_product(op::tanh(x - 3.0), op::tanh(y - 2.0));
    EXPECT_TRUE(Array::allclose(outer, expected_outer, 1e-5));
}

TEST(ArrayTests, outer_product_chainable_with_sum) {
    Array x = op::arange(3);
    Array y = op::arange(16).reshape({4, 4});
    Array outer = op::outer(x, op::sum(y, {0}));
    auto expected_outer = reference_outer_product(x, op::sum(y, {0}));
    EXPECT_TRUE(Array::allclose(outer, expected_outer, 1e-5));
}

namespace {
    Array reference_one_hot(Array indices, int depth, double on_value, double off_value) {
        auto out_shape = indices.shape();
        out_shape.emplace_back(depth);
        auto res = Array::zeros(out_shape, DTYPE_DOUBLE);
        res = res.reshape({-1, depth});
        indices = indices.ravel();
        res = off_value;
        for (int i = 0; i < indices.number_of_elements(); i++) {
            (res[i][(int)indices[i]] = on_value).eval();
        }
        return res.reshape(out_shape);
    }
}


TEST(ArrayTests, one_hot) {
    auto a = op::uniform(0, 6, {2, 3});
    EXPECT_TRUE(Array::equals(reference_one_hot(a, 7, 112.2, 42.0),
                              op::one_hot(a, 7, 112.2, 42.0)));
}


#define DALI_DEFINE_REFERENCE_UNARY(FUNCNAME, FUNCTOR_NAME)\
    Array reference_ ##FUNCNAME (Array x) {\
        Array out = Array::zeros_like(x);\
        auto raveled_x = x.ravel();\
        auto raveled_out = out.ravel();\
        if (x.dtype() == DTYPE_DOUBLE) {\
            for (int i = 0; i < raveled_x.number_of_elements(); i++) {\
                (raveled_out(i) = functor::FUNCTOR_NAME<double>::Map((double)raveled_x(i))).eval();\
            }\
        } else if (x.dtype() == DTYPE_FLOAT) {\
            for (int i = 0; i < raveled_x.number_of_elements(); i++) {\
                (raveled_out(i) = functor::FUNCTOR_NAME<float>::Map((float)raveled_x(i))).eval();\
            }\
        } else {\
            for (int i = 0; i < raveled_x.number_of_elements(); i++) {\
                (raveled_out(i) = functor::FUNCTOR_NAME<int>::Map((int)raveled_x(i))).eval();\
            }\
        }\
        return out;\
    }

#define DALI_JIT_UNARY_TEST(funcname)\
    TEST(JITTests, unary_ ##funcname) {\
        for (auto dtype : {DTYPE_INT32, DTYPE_FLOAT, DTYPE_DOUBLE}) {\
            int size = 10;\
            auto a = op::arange(5 * size).reshape({5, size}).astype(dtype) + 1;\
            auto dst = Array::zeros({5, size}, dtype);\
            dst = op::funcname(a);\
            EXPECT_TRUE(Array::allclose(dst, reference_ ##funcname(a), 1e-4));\
        }\
    }


DALI_DEFINE_REFERENCE_UNARY(softplus, softplus);
DALI_DEFINE_REFERENCE_UNARY(sigmoid, sigmoid);
DALI_DEFINE_REFERENCE_UNARY(tanh, tanh);
DALI_DEFINE_REFERENCE_UNARY(log, log);
DALI_DEFINE_REFERENCE_UNARY(cube, cube);
DALI_DEFINE_REFERENCE_UNARY(sqrt, sqrt_f);
DALI_DEFINE_REFERENCE_UNARY(rsqrt, rsqrt);
DALI_DEFINE_REFERENCE_UNARY(eltinv, inv);
DALI_DEFINE_REFERENCE_UNARY(relu, relu);
DALI_DEFINE_REFERENCE_UNARY(abs, abs);
DALI_DEFINE_REFERENCE_UNARY(sign, sign);
DALI_DEFINE_REFERENCE_UNARY(identity, identity);


DALI_JIT_UNARY_TEST(softplus);
DALI_JIT_UNARY_TEST(sigmoid);
DALI_JIT_UNARY_TEST(tanh);
DALI_JIT_UNARY_TEST(log);
DALI_JIT_UNARY_TEST(cube);
DALI_JIT_UNARY_TEST(sqrt);
DALI_JIT_UNARY_TEST(rsqrt);
DALI_JIT_UNARY_TEST(eltinv);
DALI_JIT_UNARY_TEST(relu);
DALI_JIT_UNARY_TEST(abs);
DALI_JIT_UNARY_TEST(sign);
DALI_JIT_UNARY_TEST(identity);

#define DALI_DEFINE_REFERENCE_UNARY_SCALAR(FUNCNAME, FUNCTOR_NAME)\
    Array reference_scalar_ ##FUNCNAME (Array x, double scalar) {\
        Array out = Array::zeros_like(x);\
        auto raveled_x = x.ravel();\
        auto raveled_out = out.ravel();\
        if (x.dtype() == DTYPE_DOUBLE) {\
            for (int i = 0; i < raveled_x.number_of_elements(); i++) {\
                (raveled_out(i) = functor::FUNCTOR_NAME<double>::Map((double)raveled_x(i), scalar)).eval();\
            }\
        } else if (x.dtype() == DTYPE_FLOAT) {\
            for (int i = 0; i < raveled_x.number_of_elements(); i++) {\
                (raveled_out(i) = functor::FUNCTOR_NAME<float>::Map((float)raveled_x(i), (float)scalar)).eval();\
            }\
        } else {\
            for (int i = 0; i < raveled_x.number_of_elements(); i++) {\
                (raveled_out(i) = functor::FUNCTOR_NAME<int>::Map((int)raveled_x(i), (int)scalar)).eval();\
            }\
        }\
        return out;\
    }

#define DALI_JIT_SCALAR_UNARY_TEST(funcname)\
    TEST(JITTests, scalar_unary_ ##funcname) {\
        for (auto dtype : {DTYPE_INT32, DTYPE_FLOAT, DTYPE_DOUBLE}) {\
            int size = 10;\
            auto a = op::arange(5 * size).reshape({5, size}).astype(dtype) + 1;\
            auto dst = Array::zeros({5, size}, dtype);\
            dst = op::funcname(a, 2.0);\
            EXPECT_TRUE(Array::allclose(dst, reference_scalar_ ##funcname(a, 2.0), 1e-3));\
        }\
    }

DALI_DEFINE_REFERENCE_UNARY_SCALAR(add, add);
DALI_DEFINE_REFERENCE_UNARY_SCALAR(subtract, subtract);
DALI_DEFINE_REFERENCE_UNARY_SCALAR(eltmul, eltmul);
DALI_DEFINE_REFERENCE_UNARY_SCALAR(eltdiv, eltdiv);
DALI_DEFINE_REFERENCE_UNARY_SCALAR(pow, power);

DALI_JIT_SCALAR_UNARY_TEST(add);
DALI_JIT_SCALAR_UNARY_TEST(subtract);
DALI_JIT_SCALAR_UNARY_TEST(eltmul);
DALI_JIT_SCALAR_UNARY_TEST(eltdiv);
DALI_JIT_SCALAR_UNARY_TEST(pow);


TEST(ReducerTests, all_reduce_sum) {
    int rows = 5, cols = 10;
    auto a = op::arange(rows*cols).reshape({rows, cols}).astype(DTYPE_FLOAT);
    // sum of n first numbers is (n * (n+1)) / 2:
    int expected_total = (a.number_of_elements() * (a.number_of_elements() - 1)) / 2;
    EXPECT_EQ(expected_total, (int)Array(op::sum(a)));
}

TEST(ReducerTests, all_reduce_prod) {
    auto a = op::arange(1, 6, 1);
    int expected_total = 1 * 2 * 3 * 4 * 5;
    EXPECT_EQ(expected_total, (int)Array(op::prod(a)));
}

TEST(ReducerTests, all_reduce_max_min) {
    auto a = op::arange(-100, 42, 1);
    int expected_max = 41, expected_min = -100;
    EXPECT_EQ(expected_max, (int)Array(op::max(a)));
    EXPECT_EQ(expected_min, (int)Array(op::min(a)));
}

TEST(ReducerTests, all_reduce_argmax_argmin) {
    auto a = op::arange(-100, 42, 1);
    int expected_argmax = 141, expected_argmin = 0;
    EXPECT_EQ(expected_argmax, (int)Array(op::argmax(a)));
    EXPECT_EQ(expected_argmin, (int)Array(op::argmin(a)));
}

TEST(ReducerTests, axis_reduce_argmax_argmin) {
    auto a = op::arange(4 * 5).reshape({4, 5});
    EXPECT_TRUE(Array::equals(Array::ones({5}, DTYPE_INT32) * 3, op::argmax(a, 0)));
    EXPECT_TRUE(Array::equals(Array::ones({4}, DTYPE_INT32) * 4, op::argmax(a, 1)));

    a = a * -1.0;

    EXPECT_TRUE(Array::equals(Array::ones({5}, DTYPE_INT32) * 3, op::argmin(a, 0)));
    EXPECT_TRUE(Array::equals(Array::ones({4}, DTYPE_INT32) * 4, op::argmin(a, 1)));
}

TEST(ReducerTests, all_reduce_argmax_argmin_4d) {
    auto a = op::arange(2*3*4*5).reshape({2, 3, 4, 5});
    int expected_argmax = 2 * 3 * 4 * 5 - 1, expected_argmin = 0;
    EXPECT_EQ(expected_argmax, (int)Array(op::argmax(a)));
    EXPECT_EQ(expected_argmin, (int)Array(op::argmin(a)));
}

TEST(ReducerTests, all_reduce_mean) {
    auto a = op::arange(1, 3, 1);
    double expected_mean = 1.5;
    EXPECT_EQ(expected_mean, (double)Array(op::mean(a)));
}

TEST(ReducerTests, all_reduce_l2_norm) {
    auto a = Array::ones({4}, DTYPE_INT32);
    EXPECT_EQ(2.0, (double)Array(op::L2_norm(a)));

    a = Array::ones({2}, DTYPE_INT32);
    EXPECT_EQ(std::sqrt(2.0), (double)Array(op::L2_norm(a)));
}

TEST(ReducerTests, all_reduce_sum_with_broadcast) {
    int rows = 5, cols = 10;
    Array a = op::arange(rows * cols).reshape({rows, cols}).astype(DTYPE_FLOAT)[Slice()][Broadcast()][Slice()];
    // sum of n first numbers is (n * (n+1)) / 2:
    int expected_total = (a.number_of_elements() * (a.number_of_elements() - 1)) / 2;
    EXPECT_EQ(expected_total, (int)Array(op::sum(a)));
}

TEST(ReducerTests, all_reduce_sum_with_strides) {
    int rows = 5, cols = 10, skip = 2, expected_total = 0, k=0;
    Array a = op::arange(rows * cols).reshape({rows, cols}).astype(DTYPE_FLOAT)[Slice()][Slice(0,cols,skip)];
    // sum of n first numbers, while skipping 1 out of 2 on last dim:
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (j % skip == 0) {
                expected_total += k++;
            } else {
                ++k;
            }
        }
    }
    EXPECT_EQ(expected_total, (int)Array(op::sum(a)));
}

TEST(ReducerTests, all_reduce_mixed_sum) {
    auto a = op::arange(3 * 4 * 5).reshape({3, 4, 5});
    Array b = op::arange(2 * 3 * 4 * 5).reshape({2, 3, 4, 5})[Slice()][Slice()][Slice(0, 4, 3)];
    int expected_result = (
        (a.number_of_elements() * (a.number_of_elements() - 1)) / 2 -
        (2 * 3 * 4 * 5 - 1) + 2.0
    );
    auto operation = 2 + (op::sum(a) - op::max(b));
    EXPECT_EQ(expected_result, (int)Array(operation));
}

TEST(ReducerTests, axis_reduce_sum_low_dim) {
    auto a = Array::ones({2, 3, 4, 5}, DTYPE_INT32);
    // same kernel is used in all these cases:
    EXPECT_TRUE(Array::equals(Array::ones({2, 3, 4}, DTYPE_INT32) * 5, op::sum(a, {-1})));
    EXPECT_TRUE(Array::equals(Array::ones({2, 3}, DTYPE_INT32) * 20, op::sum(a, {-2, -1})));
    EXPECT_TRUE(Array::equals(Array::ones({2}, DTYPE_INT32) * 60, op::sum(a, {-3, -2, -1})));
}

TEST(ReducerTests, axis_reduce_sum_high_dim) {
    auto a = Array::ones({2, 3, 4, 5}, DTYPE_INT32);
    // same kernel is used in all these cases:
    EXPECT_TRUE(Array::equals(Array::ones({3, 4, 5}, DTYPE_INT32) * 2, op::sum(a, {0})));
    EXPECT_TRUE(Array::equals(Array::ones({4, 5}, DTYPE_INT32) * 6, op::sum(a, {1, 0})));
    EXPECT_TRUE(Array::equals(Array::ones({5}, DTYPE_INT32) * 24, op::sum(a, {2, 1, 0})));
}

TEST(ReducerTests, axis_reduce_sum_middle_dim) {
    auto a = Array::ones({2, 3}, DTYPE_INT32);
    EXPECT_TRUE(Array::equals(Array::ones({3}, DTYPE_INT32) * 2, op::sum(a, {0})));
    EXPECT_TRUE(Array::equals(Array::ones({2}, DTYPE_INT32) * 3, op::sum(a, {1})));
    a = Array::ones({2, 3, 4}, DTYPE_INT32);
    EXPECT_TRUE(Array::equals(Array::ones({3, 4}, DTYPE_INT32) * 2, op::sum(a, {0})));
    EXPECT_TRUE(Array::equals(Array::ones({2, 4}, DTYPE_INT32) * 3, op::sum(a, {1})));
    EXPECT_TRUE(Array::equals(Array::ones({2, 3}, DTYPE_INT32) * 4, op::sum(a, {2})));
}

TEST(ReducerTests, lse_reduce) {
    auto a = Array::zeros({2}, DTYPE_INT32).insert_broadcast_axis(1);
    a <<= Array::ones({2, 5}, DTYPE_INT32);
    EXPECT_EQ(5, int(a[0][0]));
    EXPECT_EQ(5, int(a[1][0]));
}


namespace {
    Array reference_gather(Array source, Array indices) {
        auto result_shape = indices.shape();
        auto source_shape  = source.shape();
        result_shape.insert(result_shape.end(), source_shape.begin() + 1, source_shape.end());
        Array result(result_shape, source.dtype());
        source = source.reshape({source_shape.front(), -1});
        auto result_view = result.reshape({-1, source.shape().back()});
        for (int i = 0; i < indices.number_of_elements(); i++) {
            op::assign(result_view[i], OPERATOR_T_EQL, source[int(indices[i])]).eval();
        }
        return result;
    }
}


TEST(GatherTests, gather_simple) {
    auto indices = op::arange(5);
    auto source = op::arange(5 * 6).reshape({5, 6});
    EXPECT_TRUE(Array::equals(op::gather(source, indices), reference_gather(source, indices)));
    auto source2 = op::arange(5 * 6 * 7).reshape({5, 6, 7});
    EXPECT_TRUE(Array::equals(op::gather(source2, indices), reference_gather(source2, indices)));
}

TEST(GatherTests, gather_simple_elementwise) {
    auto indices = op::arange(5);
    auto source = op::arange(5 * 6).reshape({5, 6}).astype(DTYPE_DOUBLE);
    EXPECT_TRUE(Array::allclose(op::gather(op::sigmoid(source), indices),
                                reference_gather(op::sigmoid(source), indices),
                                1e-6));
}


namespace {
    Array reference_gather_from_rows(Array source, Array indices) {
        ASSERT2(source.ndim() == 2, "reference only supports 2D source");
        ASSERT2(indices.ndim() == 1, "reference only supports 1D indices");
        ASSERT2(indices.number_of_elements() == source.shape()[0], "wrong indices size.");
        Array result({source.shape()[0]}, source.dtype());
        for (int i = 0; i < indices.number_of_elements(); i++) {
            op::assign(result[i], OPERATOR_T_EQL, source[i][int(indices[i])]).eval();
        }
        return result;
    }
}

TEST(GatherTests, gather_from_rows_simple) {
    auto indices = op::arange(5);
    auto source = op::arange(5 * 6).reshape({5, 6});
    EXPECT_TRUE(Array::equals(op::gather_from_rows(source, indices), reference_gather_from_rows(source, indices)));

    auto source2 = op::arange(5 * 6 * 7).reshape({5, 6, 7});
    Array result_2d = op::gather_from_rows(source2, indices);
    std::vector<std::vector<int>> expected_result({
        {  0,   1,   2,   3,   4,   5,   6},
        { 49,  50,  51,  52,  53,  54,  55},
        { 98,  99, 100, 101, 102, 103, 104},
        {147, 148, 149, 150, 151, 152, 153},
        {196, 197, 198, 199, 200, 201, 202}
    });
    ASSERT_EQ(result_2d.shape(), std::vector<int>({int(expected_result.size()), int(expected_result[0].size())}));
    for (int i = 0; i < result_2d.shape()[0]; i++) {
        for (int j = 0; j < result_2d.shape()[1]; j++) {
            EXPECT_EQ(expected_result[i][j], int(result_2d[i][j]));
        }
    }
}

TEST(GatherTests, scatter_simple) {
    auto indices = Array::zeros({6}, DTYPE_INT32);
    std::vector<int> vals = {0, 0, 1, 1, 1, 2};
    for (int i = 0; i < vals.size(); i++) {
        (indices[i] = vals[i]).eval();
    }
    auto dest = Array::zeros({3}, DTYPE_INT32);
    auto gathered = dest[indices];
    ASSERT_EQ(gathered.shape(), indices.shape());
    (gathered += 1).eval();
    ASSERT_FALSE(gathered.is_buffer());
    EXPECT_EQ(2, int(dest[0]));
    EXPECT_EQ(3, int(dest[1]));
    EXPECT_EQ(1, int(dest[2]));
}

TEST(GatherTests, scatter_to_rows_simple) {
    auto indices = Array::zeros({6}, DTYPE_INT32);
    std::vector<int> vals = {0, 0, 1, 1, 1, 2};
    for (int i = 0; i < vals.size(); i++) {
        (indices[i] = vals[i]).eval();
    }
    auto dest = Array::zeros({7, 3}, DTYPE_INT32);
    dest = 42;
    // we eval here to ensure we don't try to assign to the tiled 42.
    // (in fact, we should add a rule that the assignment above be
    // non-optimizable/displaceable)
    dest.eval();
    auto gathered = dest.gather_from_rows(indices);
    ASSERT_EQ(gathered.shape(), indices.shape());
    gathered += 1;
    gathered.eval();
    ASSERT_FALSE(gathered.is_buffer());

    for (int i = 0; i < vals.size(); i++) {
        for (int j = 0; j < dest.shape()[1]; j++) {
            if (j != vals[i]) {
                EXPECT_EQ(42, int(dest[i][j]));
            } else {
                EXPECT_EQ(43, int(dest[i][j]));
            }
        }
    }
}


TEST(ArrayTests, destination_is_control_flow) {
    auto res = op::assign(op::control_dependency(
        2 + Array::zeros({2, 3}, DTYPE_INT32),
        2 + Array::zeros({2, 3}, DTYPE_INT32)), OPERATOR_T_ADD, op::control_dependency(
        2 + Array::zeros({2, 3}, DTYPE_INT32),
        2 + Array::zeros({2, 3}, DTYPE_INT32)) * 2);
    EXPECT_TRUE(Array::equals(op::jit::tile_scalar(6, {2, 3}), res));
}

TEST(JITTests, reshape_op) {
    Array a = Array::zeros({2, 3, 4}, DTYPE_INT32);
    a += op::arange(2 * 3 * 4).reshape({2, 3, 4});
    a.eval();
    Array b = Array::zeros({2 * 3 * 4}, DTYPE_INT32);
    b += op::arange(2 * 3 * 4);
    b.eval();
    ASSERT_TRUE(b.reshape({2,3,4}).is_buffer());
    ASSERT_TRUE(Array::equals(b.reshape({2,3,4}), a));
}

TEST(JITTests, repeated_op) {
    // reused Arrays can cause the graph simplification
    // process to get confused:
    Array a = 42;
    EXPECT_FALSE(Array::equals(a + a, a));
}

TEST(JITTests, nested_repeated_op) {
    auto a = op::arange(5);
    EXPECT_FALSE(Array::equals(a, a + a));
}

TEST(JITTests, nested_nested_repeated_op) {
    int size = 10;
    // single striding
    Array a = op::arange(5 * size).reshape({5, size});
    Array dst = op::arange(5 * size).reshape({5, size});
    dst -= op::add(a, a);
    EXPECT_TRUE(
        Array::equals(
            dst,
            -op::arange(5 * size).reshape({5, size})
        )
    );
}

TEST(BinaryTests, add) {
    int size = 10;

    for (auto dtype : {DTYPE_INT32, DTYPE_FLOAT, DTYPE_DOUBLE}) {
        auto a = op::arange(size).astype(dtype);
        auto b = op::arange(size).astype(dtype);
        Array dst = op::add(a, b);
        EXPECT_TRUE(Array::equals(dst, a + b));
    }

    for (auto dtype : {DTYPE_INT32, DTYPE_FLOAT, DTYPE_DOUBLE}) {
        auto a = op::arange(size).astype(dtype);
        auto b = op::arange(size).astype(dtype);
        Array dst = op::arange(size).astype(dtype) + 2;
        dst = op::add(a, b);
        EXPECT_TRUE(Array::equals(dst, a + b));
    }

    for (auto dtype : {DTYPE_INT32, DTYPE_FLOAT, DTYPE_DOUBLE}) {
        auto a = op::arange(size).astype(dtype);
        auto b = op::arange(size).astype(dtype);
        Array dst = op::arange(size).astype(dtype) + 2;
        dst += op::add(a, b);
        EXPECT_TRUE(
            Array::equals(
                dst,
                op::arange(size).astype(dtype) + 2 + a + b
            )
        );
    }

    for (auto dtype : {DTYPE_INT32, DTYPE_FLOAT, DTYPE_DOUBLE}) {
        auto a = op::arange(size).astype(dtype);
        auto b = op::arange(size).astype(dtype);
        Array dst = op::arange(size).astype(dtype) + 2;
        dst -= op::add(a, b);
        EXPECT_TRUE(
            Array::equals(
                dst,
                op::arange(size).astype(dtype) + 2 - (a + b)
            )
        );
    }

    for (auto dtype : {DTYPE_INT32, DTYPE_FLOAT, DTYPE_DOUBLE}) {
        auto a = op::arange(size).astype(dtype);
        auto b = op::arange(size).astype(dtype);
        Array dst = op::arange(size).astype(dtype) + 2;
        dst *= op::add(a, b);
        EXPECT_TRUE(
            Array::equals(
                dst,
                (op::arange(size).astype(dtype) + 2) * (a + b)
            )
        );
    }
}

TEST(BinaryTests, add_strided) {
    int size = 10;
    // single striding
    for (auto dtype : {DTYPE_INT32, DTYPE_FLOAT, DTYPE_DOUBLE}) {
        auto a = op::arange(size).astype(dtype);
        Array b = op::arange(2 * size).astype(dtype)[Slice(0, 2*size, 2)];
        Array dst = op::arange(size).astype(dtype) + 2;
        dst -= op::add(a, b);
        EXPECT_TRUE(
            Array::equals(
                dst,
                op::arange(size).astype(dtype) + 2 - (a + b)
            )
        );
    }

    // double striding:
    for (auto dtype : {DTYPE_INT32, DTYPE_FLOAT, DTYPE_DOUBLE}) {
        Array a = op::arange(size).astype(dtype);
        Array b = op::arange(2 * size).astype(dtype)[Slice(0, 2*size, 2)];
        Array dst = (op::arange(2 * size).astype(dtype) + 2)[Slice(0, 2 * size, 2)];
        dst -= op::add(a, b);
        EXPECT_TRUE(
            Array::equals(
                dst,
                (op::arange(2 * size).astype(dtype) + 2)[Slice(0, 2 * size, 2)] - (a + b)
            )
        );
    }

    // triple striding:
    for (auto dtype : {DTYPE_INT32, DTYPE_FLOAT, DTYPE_DOUBLE}) {
        Array a = op::arange(3 * size).astype(dtype)[Slice(0, 3 * size, 3)];
        Array b = op::arange(2 * size).astype(dtype)[Slice(0, 2 * size, 2)];
        Array dst = (op::arange(2 * size).astype(dtype) + 2)[Slice(0, 2 * size, 2)];
        dst -= op::add(a, b);
        EXPECT_TRUE(
            Array::equals(
                dst,
                (op::arange(2 * size).astype(dtype) + 2)[Slice(0, 2 * size, 2)] - (a + b)
            )
        );
    }
}

TEST(BinaryTests, add_strided_nd) {
    int size = 10;
    // single striding
    for (auto dtype : {DTYPE_INT32, DTYPE_FLOAT, DTYPE_DOUBLE}) {
        Array a = op::arange(5 * size).reshape({5, size}).astype(dtype);
        Array b = op::arange(5 * 2 * size).reshape({5, 2 * size}).astype(dtype)[Slice()][Slice(0, 2*size, 2)];
        Array dst = op::arange(5 * size).reshape({5, size}).astype(dtype) + 2;
        dst -= op::add(a, b);
        EXPECT_TRUE(
            Array::equals(
                dst,
                op::arange(5 * size).reshape({5, size}).astype(dtype) + 2 - (a + b)
            )
        );
    }
}


#define DALI_DEFINE_REFERENCE_BINARY_OP(FUNCNAME, FUNCTOR_NAME)\
    Array reference_ ##FUNCNAME (Array x, Array y) {\
        Array out = Array::zeros_like(x);\
        auto raveled_x = x.ravel();\
        auto raveled_y = y.ravel();\
        auto raveled_out = out.ravel();\
        if (x.dtype() == DTYPE_DOUBLE) {\
            for (int i = 0; i < raveled_x.number_of_elements(); i++) {\
                (raveled_out(i) = functor::FUNCTOR_NAME<double>::Map((double)raveled_x(i), (double)raveled_y(i))).eval();\
            }\
        } else if (x.dtype() == DTYPE_FLOAT) {\
            for (int i = 0; i < raveled_x.number_of_elements(); i++) {\
                (raveled_out(i) = functor::FUNCTOR_NAME<float>::Map((float)raveled_x(i), (float)raveled_y(i))).eval();\
            }\
        } else {\
            for (int i = 0; i < raveled_x.number_of_elements(); i++) {\
                (raveled_out(i) = functor::FUNCTOR_NAME<int>::Map((int)raveled_x(i), (int)raveled_y(i))).eval();\
            }\
        }\
        return out;\
    }

DALI_DEFINE_REFERENCE_BINARY_OP(eltmul, eltmul);
DALI_DEFINE_REFERENCE_BINARY_OP(eltdiv, eltdiv);
DALI_DEFINE_REFERENCE_BINARY_OP(prelu, prelu);
DALI_DEFINE_REFERENCE_BINARY_OP(pow, power);
DALI_DEFINE_REFERENCE_BINARY_OP(equals, equals);

#define DALI_RTC_BINARY_TEST(funcname)\
    TEST(BinaryTests, elementwise_binary_ ##funcname) { \
        for (auto dtype : {DTYPE_INT32, DTYPE_FLOAT, DTYPE_DOUBLE}) {\
            int size = 4;\
            auto a = op::arange(2 * size).reshape({2, size}).astype(dtype);\
            auto b = op::arange(2 * size).reshape({2, size}).astype(dtype) + 1;\
            Array dst = op::arange(2 * size).reshape({2, size}).astype(dtype) + 2;\
            dst = op::funcname(a, b);\
            Array reference = reference_ ##funcname(a, b);\
            EXPECT_TRUE(Array::allclose(dst, reference, 1e-2));\
        }\
    }


DALI_RTC_BINARY_TEST(eltmul);
DALI_RTC_BINARY_TEST(eltdiv);
DALI_RTC_BINARY_TEST(prelu);
DALI_RTC_BINARY_TEST(pow);
DALI_RTC_BINARY_TEST(equals);


TEST(BinaryTests, chained_add) {
    int size = 10;
    // single striding
    for (auto dtype : {DTYPE_INT32, DTYPE_FLOAT, DTYPE_DOUBLE}) {
        auto a = op::arange(5 * size).reshape({5, size}).astype(dtype);
        Array b = op::arange(5 * 2 * size).reshape({5, 2 * size}).astype(dtype)[Slice()][Slice(0, 2 * size, 2)];
        Array c = op::arange(5 * 3 * size).reshape({5, 3 * size}).astype(dtype)[Slice()][Slice(0, 3 * size, 3)];
        Array dst = op::arange(5 * size).reshape({5, size}).astype(dtype) + 2;
        // these two additions are a single kernel:
        dst -= op::add(op::add(a, b), c);
        EXPECT_TRUE(
            Array::equals(
                dst,
                op::arange(5 * size).reshape({5, size}).astype(dtype) + 2 - (a + b + c)
            )
        );
    }
}

TEST(BinaryTests, cast_binary) {
    // auto casts to the right type before adding:
    for (auto dtype : {DTYPE_INT32, DTYPE_FLOAT, DTYPE_DOUBLE}) {
        Array res = op::add(op::arange(10).astype(dtype), op::arange(10).astype(DTYPE_INT32));
        EXPECT_EQ(dtype, res.dtype());
        EXPECT_TRUE(Array::allclose(op::arange(10).astype(dtype) * 2, res, 1e-8));
    }
}
