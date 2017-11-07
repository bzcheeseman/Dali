#include "dot.h"

#include "dali/config.h"

#include "dali/array/op2/cpu_gemm.h"
#ifdef DALI_USE_CUDA
    #include "dali/array/op2/cublas_gemm.h"
    #include "dali/array/op2/nervana_gemm.h"
#endif

#include "dali/utils/assert2.h"
#include "dali/utils/make_message.h"


namespace expression {
    struct DotExpressionNode: public RValue {
        std::shared_ptr<const RValue> left_;
        std::shared_ptr<const RValue> right_;

        DotExpressionNode(std::shared_ptr<const RValue> left, std::shared_ptr<const RValue> right)
            : left_(left), right_(right) {
        }

        virtual std::string name() const {
            return "dot";
        }

        DType dtype() const {
            return left_->dtype();
        }

        std::vector<int> bshape() const {
            std::vector<int> result = {1, 1};
            auto left_bshape = left_->bshape();
            result[0] = left_bshape[0];
            auto right_bshape = right_->bshape();
            result[1] = right_bshape[1];
            return result;
        }

        int ndim() const {
            return 2;
        }

        std::vector<std::shared_ptr<const ExpressionNode>> arguments() const {
            return {left_, right_};
        }

        static std::tuple<double, double> operator_to_multipliers(OPERATOR_T operator_t) {
            if (operator_t == OPERATOR_T_EQL) {
                return std::make_tuple(1.0, 0.0);
            } else if (operator_t == OPERATOR_T_ADD) {
                return std::make_tuple(1.0, 1.0);
            } else if (operator_t == OPERATOR_T_SUB) {
                return std::make_tuple(-1.0, 1.0);
            } else {
                ASSERT2(false, "no multipliers available for this operator.");
            }
        }

        virtual std::shared_ptr<const expression::Runnable> use_operator(std::shared_ptr<const expression::LValue> dest,
                                                                         memory::Device device,
                                                                         OPERATOR_T operator_t) const {
            auto left_runnable  = left_->as_runnable(device);
            auto right_runnable = right_->as_runnable(device);

            // TODO(szymon): ensure contiguous when not transpose.
            auto dest_array = dest->as_array();
            if (device.is_cpu()) {
                if (dest_array) {
                    double result_multiplier, destination_multiplier;
                    std::tie(result_multiplier, destination_multiplier) = operator_to_multipliers(operator_t);
                    return std::make_shared<CpuGemmAssignExpressionNode>(dest_array, left_runnable, right_runnable, result_multiplier, destination_multiplier);
                } else {
                    return dest->operator_from(operator_t, this->as_runnable(device), device);
                }
            }
#ifdef DALI_USE_CUDA
            else if (device.is_gpu()) {
                if (dest_array) {
                    double result_multiplier, destination_multiplier;
                    std::tie(result_multiplier, destination_multiplier) = operator_to_multipliers(operator_t);

                    if (compatible_with_nervana(dtype(), device)) {
                        return std::make_shared<NervanaGemmAssignExpressionNode>(
                            dest_array,
                            left_runnable,
                            right_runnable,
                            result_multiplier,
                            destination_multiplier,
                            device
                        );
                    } else {
                        return std::make_shared<CublasGemmAssignExpressionNode>(
                            dest_array,
                            left_runnable,
                            right_runnable,
                            result_multiplier,
                            destination_multiplier,
                            device
                        );
                    }
                } else {
                    return dest->operator_from(operator_t, this->as_runnable(device), device);
                }
            }
#endif
            else {
                ASSERT2(false, "unrecognized device.");
            }
        }

        virtual std::shared_ptr<const expression::Runnable> assign_to(std::shared_ptr<const expression::LValue> dest, memory::Device device) const {
            return use_operator(dest, device, OPERATOR_T_EQL);
        }

        virtual std::shared_ptr<const expression::Runnable> plus_to(std::shared_ptr<const expression::LValue> dest, memory::Device device) const {
            return use_operator(dest, device, OPERATOR_T_ADD);
        }

        virtual std::shared_ptr<const expression::Runnable> sub_to(std::shared_ptr<const expression::LValue> dest, memory::Device device) const {
            return use_operator(dest, device, OPERATOR_T_SUB);
        }
    };

}  // namespace expression

namespace op {
    expression::ExpressionGraph dot2(const expression::ExpressionGraph& left, const expression::ExpressionGraph& right) {
        ASSERT2(left.ndim() == 2 && right.ndim() == 2, utils::make_message(
            "Inputs to dot must be two-dimensional (got left.shape() = ",
            left.shape(), ", right.shape() = ", right.shape(), ")."));
        auto left_rvalue  = left.state_->as_rvalue();
        auto right_rvalue = right.state_->as_rvalue();
        ASSERT2(left_rvalue, "First argument to dot must be a rvalue.");
        ASSERT2(right_rvalue, "Second argument to dot must be a rvalue.");
        // TODO(szymon): add type promotion.
        return expression::ExpressionGraph(std::make_shared<expression::DotExpressionNode>(left_rvalue, right_rvalue));
    }
}  // namespace op
