#include <gtest/gtest.h>

#include "dali/utils/print_utils.h"
#include "dali/runtime_config.h"
#define DALI_USE_LAZY 0
#include "dali/array/op.h"

using namespace op;

template<typename T>
void EXPECT_EQ_DTYPE(const T& reference, const Array& result, const DType& dtype) {
    if (dtype == DTYPE_FLOAT) {
        EXPECT_EQ((float)reference, (float)result);
    } else if (dtype == DTYPE_INT32) {
        EXPECT_EQ((int)reference, (int)result);
    } else if (dtype == DTYPE_DOUBLE) {
        EXPECT_EQ((double)reference, (double)result);
    }
}

void check_dot_result(DType dtype, bool contiguous) {
    Array a = Array::ones({2, 3}, dtype);
    Array b({4, 3}, dtype);
    b = initializer::arange();
    b = contiguous ? b.transpose().ascontiguousarray() : b.transpose();

    int dali_function_computations = 0;
    auto handle = debug::dali_function_computed.register_callback([&](bool ignored) {
        dali_function_computations += 1;
    });

    Array c = op::dot(a, b);

    std::vector<float> expected = {
        3, 12, 21, 30,
        3, 12, 21, 30
    };

    for (int i = 0; i < c.number_of_elements(); i++) {
        EXPECT_EQ_DTYPE(expected[i], c(i), dtype);
    }
    // make sure that the function is lazy - no extra dali functions are run
    // during computation.
    EXPECT_EQ(1, dali_function_computations);
    debug::dali_function_computed.deregister_callback(handle);
}

void check_strided_dot_result(DType dtype) {
    Array a = Array::ones({2, 3}, dtype);
    Array b({8, 3}, dtype);
    b = 999;
    b = b[Slice(0, 8, 2)];
    for (int i = 0; i < b.number_of_elements(); ++i) {
        b(i) = i;
    }

    Array c = op::dot(a, b.transpose());

    std::vector<float> expected = {
        3, 12, 21, 30,
        3, 12, 21, 30
    };

    for (int i = 0; i < c.number_of_elements(); i++) {
        EXPECT_EQ_DTYPE(expected[i], c(i), dtype);
    }
}

TEST(ArrayDotTests, dot) {
    check_dot_result(DTYPE_FLOAT, true);
    check_dot_result(DTYPE_INT32, true);
}

TEST(ArrayDotTests, dot_T) {
    check_dot_result(DTYPE_FLOAT, false);
    check_dot_result(DTYPE_INT32, false);
}

TEST(ArrayDotTests, dot_strided) {
    check_strided_dot_result(DTYPE_FLOAT);
    check_strided_dot_result(DTYPE_INT32);
}

// test off for now, requires reshaping non contiguous
// memory (e.g. perform a copy)
TEST(ArrayDotTests, tensordot) {
    Array a = Array::ones({3, 2, 1}, DTYPE_INT32);
    Array b = Array::ones({3, 1, 2}, DTYPE_INT32);
    Array c = dot(a, b);
    // c.print();
    EXPECT_EQ(std::vector<int>({3, 2, 3, 2}), c.shape());
}
