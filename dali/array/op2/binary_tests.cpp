#include <gtest/gtest.h>

#include "dali/utils/print_utils.h"
#include "dali/array/test_utils.h"
#include "dali/runtime_config.h"
#include "dali/array/op2/binary.h"


TEST(ArrayBinary2, add) {
    auto a = Array::arange({10}, DTYPE_INT32);
    auto b = Array::arange({10}, DTYPE_INT32);
    auto dst = Array::zeros({10}, DTYPE_INT32);

    op2::add(dst, a, b);

    dst.print();
}
