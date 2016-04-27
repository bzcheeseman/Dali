#ifndef DALI_ARRAY_OP_RANDOM_H
#define DALI_ARRAY_OP_RANDOM_H

#include <vector>

#include "dali/array/array.h"
#include "dali/array/dtype.h"
#include "dali/array/memory/device.h"
#include "dali/runtime_config.h"

namespace tensor_ops {
    namespace random {
        AssignableArray gaussian(const double& mean, const double& std);
        AssignableArray uniform(const double& low, const double& high);
        AssignableArray bernoulli(const double& prob);
        AssignableArray bernoulli_normalized(const double& prob);
    } // namespace random
} // namespace tensor_ops
#endif
