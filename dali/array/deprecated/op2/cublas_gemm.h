#ifndef DALI_ARRAY_OP2_CUBLAS_GEMM_H
#define DALI_ARRAY_OP2_CUBLAS_GEMM_H

#include <memory>
#include <string>
#include <vector>
#include "dali/array/op2/cpu_gemm.h"

namespace expression {
    struct CublasGemmAssignExpressionNode : public CpuGemmAssignExpressionNode {
    	memory::Device device_;

        CublasGemmAssignExpressionNode(std::shared_ptr<const expression::ArrayWrapper> dest,
                                       std::shared_ptr<const expression::Runnable> left,
                                       std::shared_ptr<const expression::Runnable> right,
                                       double result_multiplier,
                                       double destination_multiplier,
                                       memory::Device device);
        virtual void run() const;
    };
}  // namespace expression

#endif // DALI_ARRAY_OP2_CUBLAS_GEMM_H
