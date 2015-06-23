#ifndef DALI_MAT_MATH_LAZY_PLUCK_H
#define DALI_MAT_MATH_LAZY_PLUCK_H
#ifdef DOT_NOT_COMPILE_ME


#include "mshadow/tensor.h"
#include "mshadow/expr_engine-inl.h"
#include "dali/math/LazyUtils.h"


// This will go inside LazyTensor if it is used
/* Future Lazy plucking
inline LazyTensor<
        dali_expr::PluckExpression<
            LeftType,
            DType,
            mshadow::expr::ExpInfo<LeftType>::kDim - 1>,
        dali_expr::PluckExpression<
            RightType,
            DType,
            mshadow::expr::ExpInfo<RightType>::kDim - 1>,
        DType,
        mshadow::expr::type::kChainer
        > operator[](mshadow::index_t idx) const {
    auto cpu_pluck = dali_expr::PluckExpression<LeftType, DType, mshadow::expr::ExpInfo<LeftType>::kDim - 1>(
        left,
        idx);
    auto gpu_pluck = dali_expr::PluckExpression<RightType, DType, mshadow::expr::ExpInfo<RightType>::kDim - 1>(
        right,
        idx);
*/


namespace dali_expr {
    template<typename SrcExp, typename DType, int dstdim>
    struct PluckExpression:
        public mshadow::expr::MakeTensorExp<
            PluckExpression<SrcExp, DType, dstdim>,
            SrcExp,
            dstdim,
            DType > {
        const SrcExp &src_;
        mshadow::index_t idx_;
        explicit PluckExpression(const SrcExp &src, mshadow::index_t idx)
            : src_(src), idx_(idx) {}
    };
}


namespace mshadow {
namespace expr {
    template<typename SrcExp, typename DType, int dstdim>
    struct Plan<dali_expr::PluckExpression<SrcExp, DType, dstdim>, DType> {
        public:
            explicit Plan(const dali_expr::PluckExpression<SrcExp, DType, dstdim> &e)
                : src_(MakePlan(e.src_)) {}

            MSHADOW_XINLINE DType Eval(mshadow::index_t y, mshadow::index_t x) const {
                return src_.Eval(y, x);
            }

        private:
            mshadow::expr::Plan<SrcExp, DType> src_;
    };
}
}
#endif
#endif
