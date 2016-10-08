#include <gtest/gtest.h>

#include "dali/utils/print_utils.h"
#include "dali/array/test_utils.h"
#include "dali/runtime_config.h"
#include "dali/array/op2/binary.h"
#include "dali/array/op2/unary.h"
#include "dali/array/op2/gather.h"
#include "dali/array/op.h"
#include "dali/array/op2/operation.h"


TEST(RTCTests, gather_simple) {
	auto indices = Array::arange({5}, DTYPE_INT32);
	auto source = Array::arange({5, 6}, DTYPE_INT32);
	Array(op2::gather(source, indices)).print();

	auto source2 = Array::arange({5, 6, 7}, DTYPE_INT32);
	Array(op2::gather(source2, indices)).print();
}
