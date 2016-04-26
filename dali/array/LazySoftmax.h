#ifndef DALI_ARRAY_LAZY_TRANSPOSE_SOFTMAX_H
#define DALI_ARRAY_LAZY_TRANSPOSE_SOFTMAX_H

#include "mshadow/tensor.h"
#include "dali/array/KernelizedSoftmax.h"
/*
Mshadow Lazy Softmax
--------------------

Extensions to MShadow to support assignment and lazy softmax
operations. No chaining is possible, but holding a non applied
Softmax expression is now possible with these two structs and
evaluation engine specializations.
*/

namespace dali_expr {
    template<typename EType, typename DType>
    struct SoftmaxColwiseExpression: public mshadow::expr::Exp<SoftmaxColwiseExpression<EType, DType>,
                                      DType, mshadow::expr::type::kComplex> {
        const EType &exp;
        const DType temperature;
        explicit SoftmaxColwiseExpression(const EType &e, DType _temperature) : exp(e), temperature(_temperature) {}
        inline auto T(void) const -> const mshadow::expr::TransposeExp<decltype(*this), DType> {
            return mshadow::expr::TransposeExp<decltype(*this), DType>(this->self());
        }
    };

    template<typename EType, typename DType>
    struct SoftmaxRowwiseExpression: public mshadow::expr::Exp<SoftmaxRowwiseExpression<EType, DType>,
                                      DType, mshadow::expr::type::kComplex> {
        const EType &exp;
        const DType temperature;
        explicit SoftmaxRowwiseExpression(const EType &e, DType _temperature) : exp(e), temperature(_temperature) {}
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
                                dali_expr::SoftmaxColwiseExpression< tensor_t<Device, 2, DType>, DType >,
                                DType > {
            inline static void Eval(tensor_t<Device, 2, DType> *dst,
                                    const dali_expr::SoftmaxColwiseExpression< tensor_t<Device, 2, DType>, DType > &exp) {
                tensor_ops::softmax_colwise(*dst, exp.exp, exp.temperature);
            }
        };

        template<typename SV, typename Device, typename DType, template <typename, int, typename> class tensor_t>
        struct ExpComplexEngine<SV,
                                tensor_t<Device, 2, DType>,
                                dali_expr::SoftmaxRowwiseExpression< tensor_t<Device, 2, DType>, DType >,
                                DType > {
            inline static void Eval(tensor_t<Device, 2, DType> *dst,
                                    const dali_expr::SoftmaxRowwiseExpression< tensor_t<Device, 2, DType>, DType > &exp) {
                tensor_ops::softmax_rowwise(*dst, exp.exp, exp.temperature);
            }
        };
    }
}

#endif
