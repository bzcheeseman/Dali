#ifndef DALI_MAT_MATH_LAZY_PLUCK_H
#define DALI_MAT_MATH_LAZY_PLUCK_H

#include "mshadow/tensor.h"
#include "mshadow/expr_engine-inl.h"
#include "dali/mat/math/LazyUtils.h"

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

template<typename SrcExp, typename DType, int dstdim>
struct mshadow::expr::Plan<dali_expr::PluckExpression<SrcExp, DType, dstdim>, DType> {
    public:
        explicit Plan(const dali_expr::PluckExpression<SrcExp, DType, dstdim> &e)
            : src_(MakePlan(e.src_)) {}

        MSHADOW_XINLINE DType Eval(mshadow::index_t y, mshadow::index_t x) const {
            return src_.Eval(y, x);
        }

    private:
        mshadow::expr::Plan<SrcExp, DType> src_;
};

#endif
