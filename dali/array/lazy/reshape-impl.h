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

    template<int devT, typename T, typename WrappedArrayT>
    auto to_mshadow_expr(memory::Device device, const std::vector<int>& output_shape, ArrayTransformerT<WrappedArrayT> wrap_array) const ->
            decltype(mshadow::expr::take(
                MshadowWrapper<devT, int, decltype(indices)>::wrap(indices, device, output_shape, make_transform_array<devT,int,(int)1>()),
                MshadowWrapper<devT, T, decltype(src)>::wrap(src, device, output_shape, make_transform_array_collapse_trailing<devT,T,(int)2>())
            )) {

        // TODO(szymon): Notice how this op completely ignores output shape? This can result in some catastrophic errors during broadcasting
        //               but this is just a manifestation of a bigger problem - we need to make API to dali function more flexible and simpler
        //               to use, so that automated broadcasting by default does something correct and inefficient and then we can explictily
        //               if out the cases where we do some extra thinking to gain efficiency. 
        return mshadow::expr::take(
            MshadowWrapper<devT, int, decltype(indices)>::wrap(indices, device, indices.bshape(), make_transform_array<devT,int,(int)1>()),
            MshadowWrapper<devT, T, decltype(src)>::wrap(src, device, src.bshape(), make_transform_array_collapse_trailing<devT,T,(int)2>())
        );
    }
};


template<typename SrcExp, typename IndexExp>
const int LazyTake<SrcExp, IndexExp>::evaluation_dim = 2;


template<typename SrcExp, typename IndexExp>
const bool LazyTake<SrcExp, IndexExp>::collapse_leading = false;


namespace lazy {
    template<typename SrcExp, typename IndexExp>
    LazyTake<SrcExp, IndexExp> take(const SrcExp& source, const IndexExp& indices) {
        return LazyTake<SrcExp, IndexExp>(source, indices);
    }
}  // namespace lazy
