#include "gemm_utils.h"
#include "dali/utils/assert2.h"
#include "dali/utils/print_utils.h"
#include "dali/utils/make_message.h"

// compute gemm col-major transpose + stride argument
std::tuple<bool, int> gemm_stride_transpose(const Array& array) {
    if (array.strides().size() == 0) {
        return std::make_tuple(false, array.normalized_strides()[0]);
    }
    const std::vector<int>& strides = array.strides();
    if (strides[0] == 1) {
        return std::make_tuple(true, strides[1]);
    } else if (strides[1] == 1) {
        return std::make_tuple(false, strides[0]);
    }
    ASSERT2(false, utils::make_message(
        "gemm only supports arrays with a single stride (got strides: ",
        strides, ")"));
    return std::make_tuple(false, 1);
}
