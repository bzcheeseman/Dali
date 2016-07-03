#include "dali/array/function/lazy_function.h"
#include "dali/config.h"
#include <mshadow/extension/circular_convolution.h>

template<typename ContentExp, typename ShiftExp>
struct LazyCircularConvolution : LazyFunction<LazyCircularConvolution<ContentExp, ShiftExp>, ContentExp, ShiftExp> {
    ContentExp content;
    ShiftExp shift;

    LazyCircularConvolution(const ContentExp& content_, const ShiftExp& shift_)
        : LazyFunction<LazyCircularConvolution<ContentExp, ShiftExp>, ContentExp, ShiftExp>(
            content_, shift_), content(content_), shift(shift_) {}

    static std::vector<int> lazy_output_bshape(const ContentExp& content_, const ShiftExp& shift_) {
        auto content_bshape = content_.bshape();
        auto shift_bshape = shift_.bshape();

        ASSERT2(content_bshape.size() == shift_bshape.size(),
            utils::MS() << "circular_convolution content and shift must have same dimensionality"
                        << "(got content.ndim=" << content_bshape.size()
                        << " != shift.ndim=" << shift_bshape.size() << ")."
        );

        for (int i = 0; i < content_bshape.size(); i++) {
            ASSERT2(content_bshape[i] == shift_bshape[i] || content_bshape[i] == -1 || shift_bshape[i] == -1,
                utils::MS() << "circular_convolution content and shift have different size at dimension " << i
                            << "(got content.shape[" << i << "]=" << content_bshape[i] << " vs. "
                            << "shift.shape[" << i << "]=" << shift_bshape[i] << ")."
            );
            // out bshape becomes:
            // (-1, -1) => -1,
            // (-1, x) => x,
            // (x, x) => x
            content_bshape[i] = std::max(content_bshape[i], shift_bshape[i]);
        }
        return content_bshape;
    }

    template<int devT, typename T, int ndim>
    auto to_mshadow_expr(memory::Device device, const std::vector<int>& output_shape, const lazy::EvaluationSpec<devT, T, ndim>& wrap_array) const ->
            decltype(mshadow::expr::circular_convolution(
                MshadowWrapper<devT,T,ContentExp>::wrap(
                    content, device, output_shape, wrap_array
                ),
                MshadowWrapper<devT,T,ContentExp>::wrap(
                    shift, device, output_shape, wrap_array
                )
            )) {
        // TODO ensure bshape is passed correctly (with broadcasting rules)
        return mshadow::expr::circular_convolution(
            MshadowWrapper<devT,T,ContentExp>::wrap(
                content, device, output_shape, wrap_array
            ),
            MshadowWrapper<devT,T,ShiftExp>::wrap(
                shift, device, output_shape, wrap_array
            )
        );
    }
};

namespace lazy {
    template<typename ContentExp, typename ShiftExp>
    LazyCircularConvolution<ContentExp, ShiftExp> circular_convolution(
            const ContentExp& content,
            const ShiftExp& shift) {
        return LazyCircularConvolution<ContentExp, ShiftExp>(content, shift);
    }
}  // namespace lazy
