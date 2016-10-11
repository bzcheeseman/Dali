#include "rmsprop.h"

#include "dali/array/op2/binary.h"
#include "dali/array/op2/unary.h"
#include "dali/array/op_overload/common.h"
#include "dali/array/op_overload/nonlazy.h"
#include "dali/utils/assert2.h"

namespace tensor_ops {
    void rmsprop_update(Tensor& param,
                        Array& cache,
                        const double& decay_rate,
                        const double& step_size,
                        const double& smooth_eps) {
        ASSERT2(cache.number_of_elements() == param.number_of_elements(),
            utils::MS() << "cache parameter in rmsprop_update has different "
                        << "size than parameter (got " << cache.number_of_elements()
                        << " and expected " << param.number_of_elements() << ")."
        );
        cache = (
            decay_rate * cache +
            (1.0 - decay_rate) * op::square(param.dw)
        );
        // update gradient using RMSprop rule
        // DEBUG_ASSERT_POSITIVE((s.array() + smooth_eps).matrix());
        param.w -= step_size * param.dw / op::sqrt(cache + smooth_eps);
    }

    // Based on the "Generating Sequences With
    // Recurrent Neural Networks" paper:
    //     http://arxiv.org/pdf/1308.0850v5.pdf
    void rmsprop_momentum_update(Tensor& param,
                                 Array& n_cache,
                                 Array& g_cache,
                                 Array& momentum_cache,
                                 const double& decay_rate,       // eq. 42
                                 const double& momentum,         // eq. 43
                                 const double& step_size,        // eq. 44
                                 const double& smooth_eps) {     // eq. 45
        ASSERT2(n_cache.number_of_elements() == param.number_of_elements(),
            utils::MS() << "n_cache parameter in rmsprop_momentum_update has different "
                        << "size than parameter (got " << n_cache.number_of_elements()
                        << " and expected " << param.number_of_elements() << ")."
        );
        ASSERT2(g_cache.number_of_elements() == param.number_of_elements(),
            utils::MS() << "g_cache parameter in rmsprop_momentum_update has different "
                        << "size than parameter (got " << g_cache.number_of_elements()
                        << " and expected " << param.number_of_elements() << ")."
        );
        ASSERT2(momentum_cache.number_of_elements() == param.number_of_elements(),
            utils::MS() << "momentum_cache parameter in rmsprop_momentum_update has different "
                        << "size than parameter (got " << momentum_cache.number_of_elements()
                        << " and expected " << param.number_of_elements() << ")."
        );
        n_cache = (
            decay_rate         * n_cache +
            (1.0 - decay_rate) * op::square(param.dw)
        );
        g_cache = (
            decay_rate         * g_cache +
            (1.0 - decay_rate) * param.dw
        );
        momentum_cache = (momentum * momentum_cache
            - step_size * param.dw / (
                op::sqrt(n_cache - op::square(g_cache) + smooth_eps)
            )
        );
        param.w += momentum_cache;
    }
}  // namespace tensor_ops
