#ifndef DALI_ARRAY_OP_SCAN_H
#define DALI_ARRAY_OP_SCAN_H

#include "dali/array/array.h"

namespace op {
    // perform a scan inclusive or exclusive over the last
    // dimension of a tensor.
    Array axis_scan(const Array& a,
                    const std::string& reducer_name,
                    bool inclusive);
    // use a sum reduction
    Array cumsum(const Array& a, bool inclusive=true);
    // use a product reduction
    Array cumprod(const Array& a, bool inclusive=true);
}  // namespace op

#endif
