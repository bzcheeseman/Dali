#ifndef DALI_ARRAY_OP2_NERVANA_CONV_H
#define DALI_ARRAY_OP2_NERVANA_CONV_H

#include "dali/array/op2/cublas_gemm.h"

namespace expression {
    struct NervanaGemmAssignExpressionNode : public CublasGemmAssignExpressionNode {
    	using CublasGemmAssignExpressionNode::CublasGemmAssignExpressionNode;
        virtual void run() const;
    };
    int device_major_capabilities(const memory::Device& device);
    bool compatible_with_nervana(const DType& dtype, const memory::Device& device);
}  // namespace expression

#endif // DALI_ARRAY_OP2_NERVANA_CONV_H
