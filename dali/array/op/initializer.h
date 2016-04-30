#ifndef DALI_ARRAY_OP_INITIALIZER_H
#define DALI_ARRAY_OP_INITIALIZER_H

#include <vector>

#include "dali/array/array.h"
#include "dali/array/dtype.h"
#include "dali/array/memory/device.h"
#include "dali/runtime_config.h"

namespace initializer {
    AssignableArray empty();
    AssignableArray zeros();
    AssignableArray ones();
    AssignableArray arange();
    template<typename T>
    AssignableArray fill(const T& constant);

    AssignableArray gaussian(const double& mean, const double& std);
    AssignableArray uniform(const double& low, const double& high);
    AssignableArray bernoulli(const double& prob);
    AssignableArray bernoulli_normalized(const double& prob);

    AssignableArray eye(double diag = 1.0);
    // Preinitializer is first run on the matrix and then SVD initialization
    // happens. If you are unsure what kind of preinitializer to use, then try
    // weights<R>::uniform(m), were m is the number of columns in your matrix.
    // DISCLAIMER: do not use on big matrices (like embeddings) - faster
    // version is a subject of current research.
    AssignableArray svd(AssignableArray preinitializer = gaussian(0.0, 1.0));


} // namespace initializer
#endif
