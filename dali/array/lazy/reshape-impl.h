#include <vector>

#include "dali/array/shape.h"
#include "dali/array/function/lazy_function.h"

#include <mshadow/extension/take.h>

template<typename SrcExp, typename IndexExp>
struct LazyTake : public LazyFunction<LazyTake<SrcExp, IndexExp>, SrcExp, IndexExp> {
    static const int  evaluation_dim;
    static const bool collapse_leading;

    SrcExp src;
    IndexExp indices;

    static std::vector<int> lazy_output_bshape(const SrcExp& src, const IndexExp& indices) {
        std::vector<int> outbshape = indices.bshape();
        auto src_bshape = src.bshape();
        ASSERT2(
            src_bshape.size() >= 1,
            utils::MS() << "src input to LazyTake must have dimensionality >= 1 (got src.ndim()="
                        << src_bshape.size()
                        << ").");
        outbshape.insert(outbshape.end(), src_bshape.begin() + 1, src_bshape.end());
        return outbshape;
    }

    static DType lazy_output_dtype(const SrcExp& src_, const IndexExp& indices) {
        ASSERT2(
            indices.dtype() == DTYPE_INT32,
            utils::MS() << "indices input to LazyTake must be of type int32 (got " << dtype_to_name(indices.dtype()) << ")."
        );
        return src_.dtype();
    }

    LazyTake(const SrcExp& src_, const IndexExp& indices_) :
            LazyFunction<LazyTake<SrcExp,IndexExp>, SrcExp, IndexExp>(src_, indices_),
            src(src_), indices(indices_) {
    }

    template<int devT, typename T, int ndim>
    auto to_mshadow_expr(memory::Device device,
                         const std::vector<int>& output_shape,
                         const lazy::EvaluationSpec<devT, T, ndim>& wrap_array) const ->
            decltype(mshadow::expr::take(
                MshadowWrapper<devT, int, decltype(indices)>::template wrap<1>(
                    indices,
                    device,
                    indices.bshape(),
                    lazy::EvaluationSpec<devT, int, 1>::collapse_leading()
                ),
                MshadowWrapper<devT, T, decltype(src)>::template wrap<2>(
                    src,
                    device,
                    src.bshape(),
                    wrap_array.template collapse_trailing_d<2>()
                )
            )) {

        // TODO(szymon): Notice how this op completely ignores output shape? This can result in some catastrophic errors during broadcasting
        //               but this is just a manifestation of a bigger problem - we need to make API to dali function more flexible and simpler
        //               to use, so that automated broadcasting by default does something correct and inefficient and then we can explictily
        //               if out the cases where we do some extra thinking to gain efficiency.

        // static_assert(mshadow::expr::ExpInfo<cool_type>::kDim == 1, "release the kraken");

        return mshadow::expr::take(
            MshadowWrapper<devT, int, decltype(indices)>::template wrap<1>(
                indices,
                device,
                indices.bshape(),
                lazy::EvaluationSpec<devT, int, 1>::collapse_leading()
            ),
            MshadowWrapper<devT, T, decltype(src)>::template wrap<2>(
                src,
                device,
                src.bshape(),
                wrap_array.template collapse_trailing_d<2>()
            )
        );
    }
};


template<typename SrcExp, typename IndexExp>
const int LazyTake<SrcExp, IndexExp>::evaluation_dim = 2;


template<typename SrcExp, typename IndexExp>
const bool LazyTake<SrcExp, IndexExp>::collapse_leading = false;



template<typename SrcExp, typename IndexExp>
struct LazyTakeFromRows : public LazyFunction<LazyTakeFromRows<SrcExp, IndexExp>, SrcExp, IndexExp> {
    static const int  evaluation_dim;

    typedef LazyTakeFromRows<SrcExp, IndexExp> self_t;

    SrcExp src;
    IndexExp indices;

    static std::vector<int> lazy_output_bshape(const SrcExp& src, const IndexExp& indices) {
        auto indices_bshape = indices.bshape();
        auto src_bshape     = src.bshape();
        ASSERT2(
            src_bshape.size() >= 1,
            utils::MS() << "src input to LazyTakeFromRows must have dimensionality >= 1 (got src.ndim()="
                        << src_bshape.size()
                        << ").");
        ASSERT2(src_bshape.size() == indices_bshape.size() + 1,
                utils::MS() << "Take from rows expects indices (" << indices_bshape
                            << ") to have one less dimension than source expression ( " << src_bshape << ").");
        for (int i = 0; i < indices.ndim(); ++i) {
            ASSERT2(indices_bshape[i] == -1 || src_bshape[i] == -1 || indices_bshape[i] == src_bshape[i],
                    utils::MS() << "Lazy take from rows: shapes incompatible (indices shape = "
                                << indices_bshape << ", source expression shape = " << src_bshape << ").");
        }
        return indices.bshape();
    }

    static DType lazy_output_dtype(const SrcExp& src_, const IndexExp& indices) {
        ASSERT2(
            indices.dtype() == DTYPE_INT32,
            utils::MS() << "indices input to LazyTakeFromRows must be of type int32 (got " << dtype_to_name(indices.dtype()) << ")."
        );
        return src_.dtype();
    }

    LazyTakeFromRows(const SrcExp& src_, const IndexExp& indices_) :
            LazyFunction<LazyTakeFromRows<SrcExp,IndexExp>, SrcExp, IndexExp>(src_, indices_),
            src(src_), indices(indices_) {
    }

    template<int devT, typename T, int ndim>
    auto to_mshadow_expr(memory::Device device,
                         const std::vector<int>& output_shape,
                         const lazy::EvaluationSpec<devT, T, ndim>& wrap_array) const ->
            decltype(mshadow::expr::take_from_rows(
                MshadowWrapper<devT, int, decltype(indices)>::wrap(
                    indices,
                    device,
                    indices.bshape(),
                    lazy::EvaluationSpec<devT, int, lazy::ExprNdimPlusOffset<self_t, 0, ndim>::value>::collapse_leading()
                ),
                MshadowWrapper<devT, T, decltype(src)>::wrap(
                    src,
                    device,
                    src.bshape(),
                    lazy::EvaluationSpec<devT, T, lazy::ExprNdimPlusOffset<self_t, 1, ndim + 1>::value>::collapse_leading()
                )
            )) {

        return mshadow::expr::take_from_rows(
            MshadowWrapper<devT, int, decltype(indices)>::wrap(
                indices,
                device,
                indices.bshape(),
                lazy::EvaluationSpec<devT, int, lazy::ExprNdimPlusOffset<self_t, 0, ndim>::value>::collapse_leading()
            ),
            MshadowWrapper<devT, T, decltype(src)>::wrap(
                src,
                device,
                src.bshape(),
                lazy::EvaluationSpec<devT, T, lazy::ExprNdimPlusOffset<self_t, 1, ndim + 1>::value>::collapse_leading()
            )
        );
    }
};


template<typename SrcExp, typename IndexExp>
const int LazyTakeFromRows<SrcExp, IndexExp>::evaluation_dim =
        ((lazy::LazyEvaluationDim<SrcExp>::value == lazy::EVALUATION_DIM_ANY) ?
             lazy::EVALUATION_DIM_ANY :
             ((lazy::LazyEvaluationDim<SrcExp>::value > 1) ?
                 lazy::LazyEvaluationDim<SrcExp>::value - 1 :
                 lazy::EVALUATION_DIM_ERROR
             )
        );

namespace lazy {
    template<typename SrcExp, typename IndexExp>
    LazyTake<SrcExp, IndexExp> take(const SrcExp& source, const IndexExp& indices) {
        return LazyTake<SrcExp, IndexExp>(source, indices);
    }

    template<typename SrcExp, typename IndexExp>
    LazyTakeFromRows<SrcExp, IndexExp> take_from_rows(const SrcExp& source, const IndexExp& indices) {
        return LazyTakeFromRows<SrcExp, IndexExp>(source, indices);
    }
}  // namespace lazy
