#ifndef DALI_ARRAY_LAZY_IM2COL_H
#define DALI_ARRAY_LAZY_IM2COL_H


#include "dali/array/function/expression.h"
#include "dali/array/function/function.h"
#include "dali/array/function/lazy_function.h"
#include "dali/config.h"

#include <mshadow/tensor.h>
#include <mshadow/extension/patch2col_constants.h>

template<int data_format, typename SrcExp>
struct LazyIm2Col;

template<int data_format, typename SrcExp>
struct LazyCol2Im;


template<int data_format>
std::vector<int> deduce_im2col_shape(
    const std::vector<int>& src_bshape,
    const int& filter_h_,
    const int& filter_w_,
    const int& stride_h_,
    const int& stride_w_,
    const int& dilate_h_,
    const int& dilate_w_,
    const int& padding_h_,
    const int& padding_w_
);


namespace lazy {
    template<typename SrcExp>
    LazyIm2Col<mshadow::expr::DATA_FORMAT_NHWC, SrcExp> im2col_nhwc(
        const SrcExp& source,
        int filter_h,
        int filter_w,
        int stride_h,
        int stride_w,
        int dilate_h=1,
        int dilate_w=1
    );
    template<typename SrcExp>
    LazyIm2Col<mshadow::expr::DATA_FORMAT_NCHW, SrcExp> im2col_nchw(
        const SrcExp& source,
        int filter_h,
        int filter_w,
        int stride_h,
        int stride_w,
        int dilate_h=1,
        int dilate_w=1
    );

    template<typename SrcExp>
    LazyCol2Im<mshadow::expr::DATA_FORMAT_NHWC, SrcExp> col2im_nhwc(
        const SrcExp& source,
        const std::vector<int>& image_shape,
        int filter_h,
        int filter_w,
        int stride_h,
        int stride_w,
        int dilate_h=1,
        int dilate_w=1
    );
    template<typename SrcExp>
    LazyCol2Im<mshadow::expr::DATA_FORMAT_NCHW, SrcExp> col2im_nchw(
        const SrcExp& source,
        const std::vector<int>& image_shape,
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
