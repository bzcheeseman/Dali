#ifndef DALI_ARRAY_OP2_CPU_GEMM_H
#define DALI_ARRAY_OP2_CPU_GEMM_H

#include <memory>
#include <string>
#include <vector>

namespace expression {
    struct CpuGemmAssignExpressionNode : public Runnable {
        std::shared_ptr<const expression::ArrayWrapper> dest_;
        std::shared_ptr<const expression::Runnable> left_;
        std::shared_ptr<const expression::Runnable> right_;
        double result_multiplier_;
        double destination_multiplier_;

        CpuGemmAssignExpressionNode(std::shared_ptr<const expression::ArrayWrapper> dest,
                                    std::shared_ptr<const expression::Runnable> left,
                                    std::shared_ptr<const expression::Runnable> right,
                                    double result_multiplier,
                                    double destination_multiplier);
        virtual std::string name() const;
        virtual DType dtype() const;
        virtual std::vector<int> bshape() const;
        virtual int ndim() const;
        virtual std::vector<std::shared_ptr<const ExpressionNode>> arguments() const;
        virtual void run() const;
        virtual std::shared_ptr<const ExpressionNode> destination_op() const;
    };

    std::tuple<bool, int> gemm_stride_transpose(const Array& array);
}  // namespace expression

#endif // DALI_ARRAY_OP2_CPU_GEMM_H
