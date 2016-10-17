#include "adam.h"

#include "dali/array/op2/binary.h"
#include "dali/array/op2/unary.h"
#include "dali/array/op_overload/common.h"
#include "dali/array/op_overload/nonlazy.h"
#include "dali/utils/assert2.h"

namespace tensor_ops {
    void adam_update(Tensor& param,
                     Array& m,
                     Array& v,
                     const double& b1,
                     const double& b2,
                     const double& smooth_eps,
                     const double& step_size,
                     unsigned long long epoch) {
        ASSERT2(m.number_of_elements() == param.number_of_elements(),
            utils::MS() << "m parameter in adam_update has different "
                        << "size than parameter (got " << m.number_of_elements()
                        << " and expected " << param.number_of_elements() << ")."
        );
        ASSERT2(v.number_of_elements() == param.number_of_elements(),
            utils::MS() << "v parameter in adam_update has different "
                        << "size than parameter (got " << v.number_of_elements()
                        << " and expected " << param.number_of_elements() << ")."
        );
        // this affects the learning rate:
        auto fix1 = 1.0 - std::pow(b1, epoch);
        auto fix2 = 1.0 - std::pow(b2, epoch);
        double lr_t = step_size * sqrt(fix2 / fix1);

        ASSERT2(lr_t == lr_t, "Epoch learning rate is NaN. Try changing b1 or b2.");

        // update m acculumulator
        m = (1.0 - b1) * m + b1 * param.dw;
        // update v acculumulator
        v = (1.0 - b2) * v + b2 * op::square(param.dw);

        // take gradient step
        param.w -= lr_t * (m / (op::sqrt(v) + smooth_eps));
    }

}  // namespace tensor_ops
