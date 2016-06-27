#include "im2col.h"

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
        const int& padding_w_) {
    typedef mshadow::expr::UnpackPatchToCol_DimInfo<data_format, 4> im2col_info_t;

    const int w_shape = src_bshape[im2col_info_t::w_dim];
    const int h_shape = src_bshape[im2col_info_t::h_dim];

    bool image_w_gt_patch = w_shape >= filter_w_;
    bool image_h_gt_patch = h_shape >= filter_h_;

    ASSERT2(image_w_gt_patch && image_h_gt_patch,
        utils::MS() << "Im2Col " << im2col_info_t::name()
                    << " image shape should be smaller than filter size ("
                    << "filter_h=" << filter_h_ << " vs. h_dim=" << h_shape << ", "
                    << "filter_w=" << filter_w_ << " vs. w_dim=" << w_shape << "). ");

    const int i_channel_ = src_bshape[im2col_info_t::channel_dim];
    const int i_height_  = src_bshape[im2col_info_t::h_dim] + padding_h_;
    const int i_width_   = src_bshape[im2col_info_t::w_dim] + padding_w_;
    // calculate number of batches
    const int num = src_bshape[0];
    const int o_height = (i_height_ - (dilate_h_ * (filter_h_ - 1) + 1)) / stride_h_ + 1;
    const int o_width  = (i_width_  - (dilate_w_ * (filter_w_ - 1) + 1)) / stride_w_ + 1;

    return {
        filter_h_ * filter_w_ * i_channel_,
        o_height * o_width * num
    };
}

template std::vector<int> deduce_im2col_shape<mshadow::expr::DATA_FORMAT_NCHW>(
        const std::vector<int>&, const int&, const int&, const int&, const int&, const int&, const int&, const int&, const int&);
template std::vector<int> deduce_im2col_shape<mshadow::expr::DATA_FORMAT_NHWC>(
        const std::vector<int>&, const int&, const int&, const int&, const int&, const int&, const int&, const int&, const int&);
