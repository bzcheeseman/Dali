#include "dot.h"

#include "dali/array/op2/operation.h"
#include "dali/utils/assert2.h"


struct GemmAssignOperationState: public RunnableOperationState {
    std::shared_ptr<const ArrayOperationState> dest_;
    std::shared_ptr<const RunnableOperationState> left_;
    std::shared_ptr<const RunnableOperationState> right_;
    double result_multiplier_;
    double destination_multiplier_;

    GemmAssignOperationState(std::shared_ptr<const ArrayOperationState> dest,
                             std::shared_ptr<const RunnableOperationState> left,
                             std::shared_ptr<const RunnableOperationState> right,
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

    std::vector<operation_state_ptr> arguments() const {
        return {dest_, left_, right_};
    }

    void run() const {
        // Array dest_array  = dest_->array_;
        // Array left_array  = left_->destination_op()->rvalue()->as_array()->array_;
        // Array right_array = right_->destination_op()->rvalue()->as_array()->array_;
        std::cout << "OK, guys, just go play starcraft..." << std::endl;
    }

    std::shared_ptr<const OperationState> destination_op() const {
        return dest_;
    }
};


struct DotOperationState: public RValueOperationState {
    std::shared_ptr<const RValueOperationState> left_;
    std::shared_ptr<const RValueOperationState> right_;

    DotOperationState(std::shared_ptr<const RValueOperationState> left, std::shared_ptr<const RValueOperationState> right)
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

    std::vector<operation_state_ptr> arguments() const {
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

    virtual std::shared_ptr<const RunnableOperationState> use_operator(std::shared_ptr<const LValueOperationState> dest,
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
                return std::make_shared<GemmAssignOperationState>(dest_array, left_runnable, right_runnable, result_multiplier, destination_multiplier);
            } else {
                return dest->operator_from(opreator_t, this->as_runnable(device), device);
            }
        } else {
            ASSERT2(false, "oh, snap.");
            // TODO(jonathan): implement cublas, nervana.
        }
    }

    virtual std::shared_ptr<const RunnableOperationState> assign_to(std::shared_ptr<const LValueOperationState> dest, memory::Device device) const {
        return use_operator(dest, device, OPERATOR_T_EQL);
    }

    virtual std::shared_ptr<const RunnableOperationState> plus_to(std::shared_ptr<const LValueOperationState> dest, memory::Device device) const {
        return use_operator(dest, device, OPERATOR_T_ADD);
    }

    virtual std::shared_ptr<const RunnableOperationState> sub_to(std::shared_ptr<const LValueOperationState> dest, memory::Device device) const {
        return use_operator(dest, device, OPERATOR_T_SUB);
    }
};



namespace op {
    Operation dot2(const Operation& left, const Operation& right) {
        ASSERT2(left.ndim() == 2 && right.ndim() == 2,
                "Inputs to dot must be two-dimensional.");
        auto left_rvalue  = left.state_->as_rvalue();
        auto right_rvalue = right.state_->as_rvalue();
        ASSERT2(left_rvalue, "First argument for dot must be a rvalue.");
        ASSERT2(right_rvalue, "Second argument for dot must be a rvalue.");

        // TODO(szymon): add type promotion.
        return Operation(std::make_shared<DotOperationState>(left_rvalue, right_rvalue));
    }
}  // namespace op
