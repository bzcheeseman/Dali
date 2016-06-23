#ifndef DALI_ARRAY_LAZY_IM2COL_H
#define DALI_ARRAY_LAZY_IM2COL_H

#include "dali/array/function/expression.h"

#include <mshadow/tensor.h>
#include <mshadow/extension/unpack_patch2col.h>
// #include <mshadow/extension/pack_col2patch.h>

template<int data_format, typename SrcExp>
struct LazyIm2Col;

namespace lazy {
    template<typename SrcExp>
    LazyIm2Col<mshadow::expr::UNPACK_PATCH2COL_NHWC, SrcExp> im2col_nhwc(
        const SrcExp& source,
        int filter_h,
        int filter_w,
        int stride_h,
        int stride_w,
        int dilate_h=1,
        int dilate_w=1
    );
    template<typename SrcExp>
    LazyIm2Col<mshadow::expr::UNPACK_PATCH2COL_NCHW, SrcExp> im2col_nchw(
        const SrcExp& source,
        int filter_h,
        int filter_w,
        int stride_h,
        int stride_w,
        int dilate_h=1,
        int dilate_w=1
    );
}  // namespace lazy

#include "dali/array/lazy/im2col-impl.h"

#endif
