#include "adagrad.h"

#include "dali/array/op/binary.h"
#include "dali/array/op/unary.h"
#include "dali/utils/assert2.h"

namespace tensor_ops {
    void adagrad_update(Tensor& param,
                        Array& cache,
                        const double& step_size,
                        const double& smooth_eps) {
        ASSERT2(cache.number_of_elements() == param.number_of_elements(),
            utils::MS() << "cache parameter in adagrad_update has different "
                        << "size than parameter (got " << cache.number_of_elements()
                        << " and expected " << param.number_of_elements() << ")."
        );
        // update gradient cache using decay rule:
        cache += op::square(param.dw);
        // clip the gradient to prevent explosions:
        // update gradient using RMSprop rule
        param.w -= step_size * param.dw / (op::sqrt(cache) + smooth_eps);
    }
}  // namespace tensor_ops
