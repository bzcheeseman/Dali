#include <mshadow/extension/unpack_patch2col.h>
#include <mshadow/extension/pack_col2patch.h>

namespace {
    template<int dstdim>
    mshadow::Shape<dstdim> vector2shape(const std::vector<int>& ishape) {
        mshadow::Shape<dstdim> shape;
        #pragma unroll
        for (int i = 0; i < dstdim; i++) {
            shape[i] = ishape[i];
        }
        return shape;
    }
}  // end anonymous namespace

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

        auto src_bshape = src_.bshape();
        ASSERT2_SHAPE_ND(src_bshape, 4, "Im2Col");


        return deduce_im2col_shape<data_format>(src_bshape,
                                                filter_h_,
                                                filter_w_,
                                                stride_h_,
                                                stride_w_,
                                                dilate_h_,
                                                dilate_w_,
                                                /*prepad_h=*/0,
                                                /*prepad_w=*/0);
    }

    template<int devT, typename T, int ndim>
    auto to_mshadow_expr(memory::Device device, const std::vector<int>& output_shape, const lazy::EvaluationSpec<devT, T, ndim>& wrap_array) const ->
            decltype(mshadow::expr::unpack_patch2col<data_format>(
                MshadowWrapper<devT,T,SrcExp>::wrap(
                    src, device, output_shape, wrap_array.template collapse_leading_d<4>()
                ), filter_h, filter_w, stride_h, stride_w, dilate_h, dilate_w, 0, 0, 0, 0
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
            dilate_w,
            /*prepad_h=*/0,
            /*prepad_w=*/0,
            /*postpad_h=*/0,
            /*postpad_w=*/0
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

template<int data_format, typename SrcExp>
struct LazyCol2Im : public LazyFunction<LazyCol2Im<data_format, SrcExp>, SrcExp, std::vector<int>, int, int, int, int, int, int> {
    typedef mshadow::expr::UnpackPatchToCol_DimInfo<data_format, 4> im2col_info_t;
    static const int  evaluation_dim;
    LazyCol2Im(const SrcExp& src_,
               const std::vector<int>& image_shape_,
               const int& filter_h_,
               const int& filter_w_,
               const int& stride_h_,
               const int& stride_w_,
               const int& dilate_h_,
               const int& dilate_w_)
        : LazyFunction<LazyCol2Im<data_format, SrcExp>, SrcExp, std::vector<int>, int, int, int, int, int, int>(
            src_, image_shape_, filter_h_, filter_w_, stride_h_, stride_w_, dilate_h_, dilate_w_),
          src(src_),
          image_shape(image_shape_),
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
    const std::vector<int> image_shape;

    static std::vector<int> lazy_output_bshape(const SrcExp& src_,
                                               const std::vector<int>& image_shape_,
                                               const int& filter_h_,
                                               const int& filter_w_,
                                               const int& stride_h_,
                                               const int& stride_w_,
                                               const int& dilate_h_,
                                               const int& dilate_w_) {

        auto src_bshape = src_.bshape();
        ASSERT2_SHAPE_ND(src_bshape, 2, "Col2Im");
        ASSERT2_SHAPE_ND(image_shape_, 4, "Col2Im's image_shape");

        const int w_shape = image_shape_[im2col_info_t::w_dim];
        const int h_shape = image_shape_[im2col_info_t::h_dim];

        bool image_w_gt_patch = w_shape >= filter_w_;
        bool image_h_gt_patch = h_shape >= filter_h_;

        ASSERT2(image_w_gt_patch && image_h_gt_patch,
            utils::MS() << "Col2Im " << im2col_info_t::name()
                        << " image shape should be smaller than filter size ("
                        << "filter_h=" << filter_h_ << " vs. h_dim=" << h_shape << ", "
                        << "filter_w=" << filter_w_ << " vs. w_dim=" << w_shape << "). ");

        const int i_channel_ = image_shape_[im2col_info_t::channel_dim];
        const int i_height_  = image_shape_[im2col_info_t::h_dim];
        const int i_width_   = image_shape_[im2col_info_t::w_dim];
        // calculate number of batches
        const int num = image_shape_[0];
        const int o_height = (i_height_ - (dilate_h_ * (filter_h_ - 1) + 1)) / stride_h_ + 1;
        const int o_width  = (i_width_  - (dilate_w_ * (filter_w_ - 1) + 1)) / stride_w_ + 1;

        ASSERT2(filter_h_ * filter_w_ * i_channel_ == src_bshape[0],
            utils::MS() << "Col2Im " << im2col_info_t::name()
                        << " input must have shape()[0] == "
                        << filter_h_ * filter_w_ * i_channel_
                        << " = filter_h [" << filter_h_
                        << "] * filter_w [" << filter_w_
                        << "] * i_channel [" << i_channel_
                        << "] (got " << src_bshape[0] << ")."
        );

        ASSERT2(o_height * o_width * num == src_bshape[1],
            utils::MS() << "Col2Im " << im2col_info_t::name()
                        << " input must have shape()[1] == "
                        << o_height * o_width * num
                        << " = {(i_height_ - (dilate_h_ * (filter_h_ - 1) + 1)) / stride_h_ + 1} [" << o_height
                        << "] * {(i_width_  - (dilate_w_ * (filter_w_ - 1) + 1)) / stride_w_ + 1} [" << o_width
                        << "] * batch_size [" << num
                        << "] (got " << src_bshape[1] << ")."
        );
        return image_shape_;
    }

    template<int devT, typename T, int ndim>
    auto to_mshadow_expr(memory::Device device, const std::vector<int>& output_shape, const lazy::EvaluationSpec<devT, T, ndim>& wrap_array) const ->
            decltype(mshadow::expr::pack_col2patch<data_format>(
                MshadowWrapper<devT,T,SrcExp>::wrap(
                    src, device, output_shape, wrap_array.template collapse_leading_d<2>()
                ), vector2shape<4>(image_shape), filter_h, filter_w, stride_h,
                stride_w, dilate_h, dilate_w, 0, 0, 0, 0
            )) {
        return mshadow::expr::pack_col2patch<data_format>(
            MshadowWrapper<devT,T,SrcExp>::wrap(
                src,
                device,
                src.shape(),
                wrap_array.template collapse_leading_d<2>()
            ),
            vector2shape<4>(image_shape),
            filter_h,
            filter_w,
            stride_h,
            stride_w,
            dilate_h,
            dilate_w,
            /*prepad_h=*/0,
            /*prepad_w=*/0,
            /*postpad_h=*/0,
            /*postpad_w=*/0
        );
    }
};

template<int data_format, typename SrcExp>
const int LazyCol2Im<data_format, SrcExp>::evaluation_dim =
    ((lazy::LazyEvaluationDim<SrcExp>::value == lazy::EVALUATION_DIM_ANY ||
      lazy::LazyEvaluationDim<SrcExp>::value == 4) ?
        4 :
        lazy::EVALUATION_DIM_ERROR
    );

namespace lazy {
    template<typename SrcExp>
    LazyIm2Col<mshadow::expr::DATA_FORMAT_NHWC, SrcExp> im2col_nhwc(
            const SrcExp& source,
            int filter_h,
            int filter_w,
            int stride_h,
            int stride_w,
            int dilate_h,
            int dilate_w) {
        return LazyIm2Col<mshadow::expr::DATA_FORMAT_NHWC, SrcExp>(
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
    LazyIm2Col<mshadow::expr::DATA_FORMAT_NCHW, SrcExp> im2col_nchw(
            const SrcExp& source,
            int filter_h,
            int filter_w,
            int stride_h,
            int stride_w,
            int dilate_h,
            int dilate_w) {
        return LazyIm2Col<mshadow::expr::DATA_FORMAT_NCHW, SrcExp>(
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
    LazyCol2Im<mshadow::expr::DATA_FORMAT_NHWC, SrcExp> col2im_nhwc(
            const SrcExp& source,
            const std::vector<int>& image_shape,
            int filter_h,
            int filter_w,
            int stride_h,
            int stride_w,
            int dilate_h,
            int dilate_w) {
        return LazyCol2Im<mshadow::expr::DATA_FORMAT_NHWC, SrcExp>(
            source,
            image_shape,
            filter_h,
            filter_w,
            stride_h,
            stride_w,
            dilate_h,
            dilate_w
        );
    }

    template<typename SrcExp>
    LazyCol2Im<mshadow::expr::DATA_FORMAT_NCHW, SrcExp> col2im_nchw(
            const SrcExp& source,
            const std::vector<int>& image_shape,
            int filter_h,
            int filter_w,
            int stride_h,
            int stride_w,
            int dilate_h,
            int dilate_w) {
        return LazyCol2Im<mshadow::expr::DATA_FORMAT_NCHW, SrcExp>(
            source,
            image_shape,
            filter_h,
            filter_w,
            stride_h,
            stride_w,
            dilate_h,
            dilate_w
        );
    }
}  // namespace lazy
