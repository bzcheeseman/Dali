#ifndef DALI_ARRAY_OP2_NERVANA_CONV_H
#define DALI_ARRAY_OP2_NERVANA_CONV_H

#include "dali/array/op2/expression/expression.h"
#include "dali/array/op2/cublas_gemm.h"

namespace expression {
    struct NervanaGemmAssignExpressionState : public CublasGemmAssignExpressionState {
    	using CublasGemmAssignExpressionState::CublasGemmAssignExpressionState;
        virtual void run() const;
    };
    int device_major_capabilities(const memory::Device& device);
    bool device_compatible_with_nervana(const memory::Device& device);
}  // namespace expression

#endif // DALI_ARRAY_OP2_NERVANA_CONV_H
