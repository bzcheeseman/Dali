#ifndef DALI_ARRAY_TESTS_TEST_UTILS_H
#define DALI_ARRAY_TESTS_TEST_UTILS_H
#include <gtest/gtest.h>

#include "dali/array/array.h"
#include "dali/array/debug.h"
#include "dali/array/dtype.h"
#include "dali/utils/observer.h"

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

template<typename T>
void EXPECT_NEAR_DTYPE(const T& reference, const Array& result, float eps, const DType& dtype) {
    if (dtype == DTYPE_FLOAT) {
        EXPECT_NEAR((float)reference, (float)result, (float)eps);
    } else if (dtype == DTYPE_INT32) {
        EXPECT_NEAR((int)reference, (int)result, eps);
    } else if (dtype == DTYPE_DOUBLE) {
        EXPECT_NEAR((double)reference, (double)result, (double)eps);
    }
}


using namespace std::placeholders;

struct CountImplicitCopies {
    ObserverGuard<Array> scoped_dc;
    int count;

    void mark_copy(const Array& arr) {
        count += 1;
    }

    CountImplicitCopies() :
            count(0),
            scoped_dc(make_observer_guard(
                    // std::bind(&CountImplicitCopies::mark_copy, this, _1),
                    [this](const Array& arr) { this->count += 1; },
                    &debug::array_as_contiguous)) {
    }
};

namespace {
    template<typename T>
    void assign_from_vec(Array out, const std::vector<std::vector<T>>& vec) {
        for (int i = 0; i < vec.size(); i++) {
            for (int j = 0; j < vec[i].size(); j++) {
                out[i][j].assign(vec[i][j]).eval();
            }
        }
    }
}

#endif