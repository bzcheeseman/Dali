#include "dot.h"

extern "C" {
    #include <cblas.h>
}

#include "dali/array/op2/expression/expression.h"
#include "dali/utils/assert2.h"
#include "dali/utils/make_message.h"
#include "dali/array/mshadow_extension/reference_gemm.h"

// support integers in gemm
void igemm(bool transpose_a, bool transpose_b,
          size_t m, size_t n, size_t k, double alpha, const int* a,
          size_t lda, const int* b, size_t ldb, double beta, int* c, size_t ldc) {
    ReferenceGemm<int>(
        transpose_a, transpose_b, true, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc
    );
}

// compute gemm col-major transpose + stride argument
std::tuple<bool, int> gemm_stride_transpose(const Array& array) {
    if (array.strides().size() == 0) {
        return std::make_tuple(false, array.normalized_strides()[0]);
    }
    const std::vector<int>& strides = array.strides();
    if (strides[0] == 1) {
        return std::make_tuple(true, strides[1]);
    } else if (strides[1] == 1) {
        return std::make_tuple(false, strides[0]);
    } else {
        ASSERT2(false, utils::make_message("gemm "
            "only supports arrays with a single stride (array strides: ",
            strides, ")"));
        return std::make_tuple(false, 1);
    }
}

struct GemmAssignExpressionState: public Runnable {
    std::shared_ptr<const ArrayWrapper> dest_;
    std::shared_ptr<const Runnable> left_;
    std::shared_ptr<const Runnable> right_;
    double result_multiplier_;
    double destination_multiplier_;

    GemmAssignExpressionState(std::shared_ptr<const ArrayWrapper> dest,
                             std::shared_ptr<const Runnable> left,
                             std::shared_ptr<const Runnable> right,
                             double result_multiplier,
                             double destination_multiplier)
            : dest_(dest),
              left_(left),
              right_(right),
              result_multiplier_(result_multiplier),
              destination_multiplier_(destination_multiplier) {
    }

    virtual std::string name() const {
        return "gemm";
    }

    DType dtype() const {
        return left_->dtype();
    }

    std::vector<int> bshape() const {
        return dest_->bshape();
    }

    int ndim() const {
        return 2;
    }

    std::vector<std::shared_ptr<const ExpressionState>> arguments() const {
        return {dest_, left_, right_};
    }

    void run() const {
        // TODO(szymon): make less brittle (please :)
        Array dst = dest_->array_;
        Array lhs = left_->destination_op()->as_rvalue()->as_array()->array_;
        Array rhs = right_->destination_op()->as_rvalue()->as_array()->array_;
        auto op_dtype = dtype();
        auto device = memory::Device::cpu();
        void* dst_ptr = destination_multiplier_ == 0 ?
            dst.memory()->overwrite_data(device) : dst.memory()->mutable_data(device);
        const void* rhs_ptr = rhs.memory()->readonly_data(device);
        const void* lhs_ptr = lhs.memory()->readonly_data(device);
        bool rhs_transpose, lhs_transpose, dst_transpose;
        int rhs_stride, lhs_stride, dst_stride;
        std::tie(rhs_transpose, rhs_stride) = gemm_stride_transpose(rhs);
        std::tie(lhs_transpose, lhs_stride) = gemm_stride_transpose(lhs);
        std::tie(dst_transpose, dst_stride) = gemm_stride_transpose(dst);
        // in row major:
        // dst = result_multiplier * left * right + destination_multiplier * dst
        // in col major:
        // dst.T = result_multiplier * right.T * left.T + destination_multiplier * dst.T
        int m = rhs.shape()[1],
            n = lhs.shape()[0],
            k = rhs.shape()[0];

        if (op_dtype == DTYPE_INT32) {
            igemm(
                rhs_transpose, lhs_transpose,
                m, n, k,
                /*alpha=*/result_multiplier_,
                (const int*)rhs_ptr, rhs_stride,
                (const int*)lhs_ptr, lhs_stride,
                /*beta=*/destination_multiplier_,
                (int*)dst_ptr, dst_stride
            );
        } else if (op_dtype == DTYPE_FLOAT) {
            cblas_sgemm(CblasColMajor,
                rhs_transpose ? CblasTrans : CblasNoTrans, lhs_transpose ? CblasTrans : CblasNoTrans,
                m, n, k,
                /*alpha=*/result_multiplier_,
                (const float*)rhs_ptr, rhs_stride,
                (const float*)lhs_ptr, lhs_stride,
                /*beta=*/destination_multiplier_,
                (float*)dst_ptr, dst_stride
            );
        } else if (op_dtype == DTYPE_DOUBLE) {
            cblas_dgemm(CblasColMajor,
                rhs_transpose ? CblasTrans : CblasNoTrans, lhs_transpose ? CblasTrans : CblasNoTrans,
                m, n, k,
                /*alpha=*/result_multiplier_,
                (const double*)rhs_ptr, rhs_stride,
                (const double*)lhs_ptr, lhs_stride,
                /*beta=*/destination_multiplier_,
                (double*)dst_ptr, dst_stride
            );
        } else {
            ASSERT2(false, utils::make_message("gemm only supports "
                DALI_ACCEPTABLE_DTYPE_STR " (got dtype=", op_dtype, ")."));
        }
    }

    std::shared_ptr<const ExpressionState> destination_op() const {
        return dest_;
    }
};


struct DotExpressionState: public RValueExpressionState {
    std::shared_ptr<const RValueExpressionState> left_;
    std::shared_ptr<const RValueExpressionState> right_;

    DotExpressionState(std::shared_ptr<const RValueExpressionState> left, std::shared_ptr<const RValueExpressionState> right)
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

    std::vector<std::shared_ptr<const ExpressionState>> arguments() const {
        return {left_, right_};
    }

    static std::tuple<double, double> operator_to_multipliers(OPERATOR_T opreator_t) {
        if (opreator_t == OPERATOR_T_EQL) {
            return std::make_tuple(1.0, 0.0);
        } else if (opreator_t == OPERATOR_T_ADD) {
            return std::make_tuple(1.0, 1.0);
        } else if (opreator_t == OPERATOR_T_SUB) {
            return std::make_tuple(-1.0, 1.0);
        } else {
            ASSERT2(false, "no multipliers available for this operator.");
        }
    }

    virtual std::shared_ptr<const Runnable> use_operator(std::shared_ptr<const LValueExpressionState> dest,
                                                                                memory::Device device,
                                                                                OPERATOR_T opreator_t) const {
        if (device.is_cpu()) {
            auto left_runnable  = left_->as_runnable(device);
            auto right_runnable = right_->as_runnable(device);

            // TODO(szymon): ensure conriguous when not transpose.

            auto dest_array = dest->as_array();
            if (dest_array) {
                double result_multiplier, destination_multiplier;
                std::tie(result_multiplier, destination_multiplier) = operator_to_multipliers(opreator_t);
                return std::make_shared<GemmAssignExpressionState>(dest_array, left_runnable, right_runnable, result_multiplier, destination_multiplier);
            } else {
                return dest->operator_from(opreator_t, this->as_runnable(device), device);
            }
        } else {
            ASSERT2(false, "oh, snap.");
            // TODO(jonathan): implement cublas, nervana.
        }
    }

    virtual std::shared_ptr<const Runnable> assign_to(std::shared_ptr<const LValueExpressionState> dest, memory::Device device) const {
        return use_operator(dest, device, OPERATOR_T_EQL);
    }

    virtual std::shared_ptr<const Runnable> plus_to(std::shared_ptr<const LValueExpressionState> dest, memory::Device device) const {
        return use_operator(dest, device, OPERATOR_T_ADD);
    }

    virtual std::shared_ptr<const Runnable> sub_to(std::shared_ptr<const LValueExpressionState> dest, memory::Device device) const {
        return use_operator(dest, device, OPERATOR_T_SUB);
    }
};



namespace op {
    Expression dot2(const Expression& left, const Expression& right) {
        ASSERT2(left.ndim() == 2 && right.ndim() == 2,
                "Inputs to dot must be two-dimensional.");
        auto left_rvalue  = left.state_->as_rvalue();
        auto right_rvalue = right.state_->as_rvalue();
        ASSERT2(left_rvalue, "First argument for dot must be a rvalue.");
        ASSERT2(right_rvalue, "Second argument for dot must be a rvalue.");
        // TODO(szymon): add type promotion.
        return Expression(std::make_shared<DotExpressionState>(left_rvalue, right_rvalue));
    }
}  // namespace op
