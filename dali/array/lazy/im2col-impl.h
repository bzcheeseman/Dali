template<int data_format, typename SrcExp>
struct LazyIm2Col : public LazyFunction<LazyIm2Col<data_format, SrcExp>, SrcExp, int, int, int, int, int, int> {
    static const int  evaluation_dim;

    LazyIm2Col(const SrcExp& src_,
               const int& filter_h_,
               const int& filter_w_,
               const int& stride_h_,
               const int& stride_w_,
               const int& dilate_h_,
               const int& dilate_w_)
        : LazyFunction<LazyIm2Col<data_format, SrcExp>, SrcExp, int, int, int, int, int, int>(
            src_, filter_h_, filter_w_, stride_h_, stride_w_, dilate_h_, dilate_w_),
          src(src_),
          filter_h(filter_h_),
          filter_w(filter_w_),
          stride_h(stride_h_),
          stride_w(stride_w_),
          dilate_h(dilate_h_),
          dilate_w(dilate_w_) {}

    SrcExp src;
    const int filter_h;
    const int filter_w;
    const int stride_h;
    const int stride_w;
    const int dilate_h;
    const int dilate_w;

    static std::vector<int> lazy_output_bshape(const SrcExp& src_,
                                               const int& filter_h_,
                                               const int& filter_w_,
                                               const int& stride_h_,
                                               const int& stride_w_,
                                               const int& dilate_h_,
                                               const int& dilate_w_) {
        using mshadow::expr::UnpackPatchToCol_DimInfo;

        auto src_bshape = src_.bshape();
        ASSERT2_SHAPE_ND(src_bshape, 4, "LazyIm2Col");

        const int w_shape = src_bshape[UnpackPatchToCol_DimInfo<data_format, 4>::w_dim];
        const int h_shape = src_bshape[UnpackPatchToCol_DimInfo<data_format, 4>::h_dim];

        bool image_w_gt_patch = w_shape >= filter_w_;
        bool image_h_gt_patch = h_shape >= filter_h_;

        ASSERT2(image_w_gt_patch && image_h_gt_patch,
            utils::MS() << "LazyIm2Col image shape should be smaller than filter size ("
                        << "filter_h=" << filter_h_ << " vs. h_dim=" << h_shape << ", "
                        << "filter_w=" << filter_w_ << " vs. w_dim=" << w_shape << "). ");

        std::vector<int> outbshape(2, 0);

        const int i_channel_ = src_bshape[UnpackPatchToCol_DimInfo<data_format, 4>::channel_dim];
        const int i_height_  = src_bshape[UnpackPatchToCol_DimInfo<data_format, 4>::h_dim];
        const int i_width_   = src_bshape[UnpackPatchToCol_DimInfo<data_format, 4>::w_dim];
        // calculate number of batches
        const int num = src_bshape[0];
        const int o_height = (i_height_ - (dilate_h_ * (filter_h_ - 1) + 1)) / stride_h_ + 1;
        const int o_width  = (i_width_  - (dilate_w_ * (filter_w_ - 1) + 1)) / stride_w_ + 1;

        outbshape[0] = filter_h_ * filter_w_ * i_channel_;
        outbshape[1] = o_height * o_width * num;

        return outbshape;
    }

    template<int devT, typename T, int ndim>
    auto to_mshadow_expr(memory::Device device, const std::vector<int>& output_shape, const lazy::EvaluationSpec<devT, T, ndim>& wrap_array) const ->
            decltype(mshadow::expr::unpack_patch2col<data_format>(
                MshadowWrapper<devT,T,SrcExp>::wrap(
                    src, device, output_shape, wrap_array.template collapse_leading_d<4>()
                ), filter_h, filter_w, stride_h, stride_w, dilate_h, dilate_w
            )) {
        return mshadow::expr::unpack_patch2col<data_format>(
            MshadowWrapper<devT,T,SrcExp>::wrap(
                src,
                device,
                src.shape(),
                wrap_array.template collapse_leading_d<4>()
            ),
            filter_h,
            filter_w,
            stride_h,
            stride_w,
            dilate_h,
            dilate_w
        );
    }
};

template<int data_format, typename SrcExp>
const int LazyIm2Col<data_format, SrcExp>::evaluation_dim =
    ((lazy::LazyEvaluationDim<SrcExp>::value == lazy::EVALUATION_DIM_ANY ||
      lazy::LazyEvaluationDim<SrcExp>::value == 2) ?
        2 :
        lazy::EVALUATION_DIM_ERROR
    );

namespace lazy {
    template<typename SrcExp>
    LazyIm2Col<mshadow::expr::UNPACK_PATCH2COL_NHWC, SrcExp> im2col_nhwc(const SrcExp& source,
                                   int filter_h,
                                   int filter_w,
                                   int stride_h,
                                   int stride_w,
                                   int dilate_h,
                                   int dilate_w) {
        return LazyIm2Col<mshadow::expr::UNPACK_PATCH2COL_NHWC, SrcExp>(
            source,
            filter_h,
            filter_w,
            stride_h,
            stride_w,
            dilate_h,
            dilate_w
        );
    }

    template<typename SrcExp>
    LazyIm2Col<mshadow::expr::UNPACK_PATCH2COL_NCHW, SrcExp> im2col_nchw(const SrcExp& source,
                                   int filter_h,
                                   int filter_w,
                                   int stride_h,
                                   int stride_w,
                                   int dilate_h,
                                   int dilate_w) {
        return LazyIm2Col<mshadow::expr::UNPACK_PATCH2COL_NCHW, SrcExp>(
            source,
            filter_h,
            filter_w,
            stride_h,
            stride_w,
            dilate_h,
            dilate_w
        );
    }
}  // namespace lazy
