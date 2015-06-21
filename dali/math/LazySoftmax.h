#ifndef DALI_MAT_MATH_LAZY_SOFTMAX_H
#define DALI_MAT_MATH_LAZY_SOFTMAX_H

#include "mshadow/tensor.h"

namespace dali_expr {
    template<typename EType, typename DType>
    struct SoftmaxExpression: public mshadow::expr::Exp<SoftmaxExpression<EType, DType>,
                                      DType, mshadow::expr::type::kComplex> {
        const EType &exp;
        explicit SoftmaxExpression(const EType &e) : exp(e) {}
        inline auto T(void) const -> const mshadow::expr::TransposeExp<decltype(*this), DType> {
            return mshadow::expr::TransposeExp<decltype(*this), DType>(this->self());
        }
    };
}

namespace mshadow {
namespace expr {
    template<typename SV, typename Device, typename DType, template <typename, int, typename> class tensor_t>
    struct ExpComplexEngine<SV,
                            tensor_t<Device, 2, DType>,
                            dali_expr::SoftmaxExpression< tensor_t<Device, 2, DType>, DType >,
                            DType > {
        inline static void Eval(tensor_t<Device, 2, DType> *dst,
                                const dali_expr::SoftmaxExpression< tensor_t<Device, 2, DType>, DType > &exp) {
            mshadow::Softmax(*dst, exp.exp);
        }
    };
}
}
#endif
