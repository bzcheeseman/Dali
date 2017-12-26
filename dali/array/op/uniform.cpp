#include "uniform.h"
#include "dali/utils/make_message.h"
#include "dali/utils/assert2.h"
#include "dali/utils/print_utils.h"
#include "dali/array/expression/computation.h"
#include "dali/array/op/elementwise_operation.h"

// DOT SPECIFIC CLASSES
namespace op {
    Uniform::Uniform(Array low, Array high, const std::vector<int>& shape) :
        Expression(shape,
                   low.dtype()),
                   low_(low), high_(high) {}
    std::vector<Array> Uniform::arguments() const {
        return {low_, high_};
    }
    expression_ptr Uniform::copy() const {
        return std::make_shared<Uniform>(*this);
    }
    memory::Device Uniform::preferred_device() const {
        return memory::default_preferred_device;
    }
    bool Uniform::supports_operator(OPERATOR_T operator_t) const {
        return operator_t == OPERATOR_T_EQL;
    }

    struct CpuUniformImpl : public Computation {
        using Computation::Computation;
        virtual void run() {
            Array dst = left_;
            op::Uniform* uni = static_cast<op::Uniform*>(right_.expression().get());
            Array low = uni->low_;
            Array high = uni->high_;
            auto op_dtype = dst.dtype();
            auto device = memory::Device::cpu();
            void* dst_ptr = dst.memory()->overwrite_data(device);
            const void* low_ptr = low.memory()->readonly_data(device);
            const void* high_ptr = high.memory()->readonly_data(device);

            // FOR FLOAT & DOUBLE
            // mshadow::Random<mshadow::cpu, T> generator(utils::randint(0,999999));
            // auto m_out = out.contiguous_d1(memory::AM_OVERWRITE);
            // generator.SampleUniform(&m_out, lower, upper);

            // FOR INTS:
            // template<OPERATOR_T operator_t, typename T, DALI_FUNC_ENABLE_IF_INT>
            // void typed_eval(TypedArray<memory::DEVICE_T_CPU, T> out, const double& lower, const double& upper) {
            //     assert_contiguous_memory(out);
            //     // uniform_int_distribution can only tak ints as per standard
            //     // clang is more permissive here.
            //     std::uniform_int_distribution<int> dist(lower, upper);
            //     auto& gen = utils::random::generator();
            //     if (operator_t == OPERATOR_T_EQL) {
            //         auto ptr = out.ptr(memory::AM_OVERWRITE);
            //         for (int i = 0; i < out.array.number_of_elements(); ++i) {
            //             *(ptr + i) = (int)dist(gen);
            //         }
            //     } else {
            //         ASSERT2(false, utils::make_message(operator_t, " not yet "
            //             "implemented for UniformInitializer"));
            //     }
            // }
        }
    };

    int cpu_uniform_impl = register_implementation(
        typeid(op::Uniform).name(),
        [](Array dest, OPERATOR_T operator_t, Array x, Array assignment) -> std::shared_ptr<Computation> {
            if (dest.preferred_device().is_cpu()) {
                return std::make_shared<CpuUniformImpl>(dest, operator_t, x, assignment);
            } else {
                return nullptr;
            }
        }
    );

    Array uniform(Array low, Array high, const std::vector<int>& shape) {
        ASSERT2(low.is_scalar(), utils::make_message(
            "low must be a scalar (got low.shape = ", low.shape(), ")."));
        ASSERT2(high.is_scalar(), utils::make_message(
            "high must be a scalar (got high.shape = ", high.shape(), ")."));
        std::tie(low, high) = ensure_arguments_compatible(low, high, "uniform");
        return Array(std::make_shared<op::Uniform>(low, high, shape));
    }

}  // namespace op
